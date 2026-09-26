"""Tri-signal retrieval: independent dense, sparse and graph channels.

Each channel ranks the same scope-local ``ScopedCorpus`` on its own evidence:

- dense  : exact (or measured-approximate HNSW) inner product over stored
           native embeddings (``engine.dense_channel``);
- sparse : BM25S, Lucene variant, scope-local IDF (``storage.bm25_index``);
- graph  : entity-memory graph from source turns with query-derived entity
           anchors, scored by the EMG entity channel or HippoRAG-2-style
           personalized PageRank (``engine.entity_graph``).

No channel seeds another unless the caller asks for it explicitly (HippoRAG 2
passage seeds), and then the dependency is recorded in the trace and the seed
channel must itself be requested. Channel lists are fused (RRF k=60 by
default), optionally reranked inside a fixed pool, and packed into an evidence
set with stable evidence IDs.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from engine.corpus import ScopedCorpus, TurnRecord

TRACE_SCHEMA = "hybridmind.trisignal-retrieval/v1"
CHANNELS = ("dense", "sparse", "graph")
FUSION_MODES = ("rrf", "dbsf", "zscore", "minmax_linear")
GRAPH_METHODS = ("emg", "ppr")
DENSE_MODES = ("exact", "hnsw")
EVIDENCE_STRATEGIES = ("turn", "window", "propagate", "session")
LEXICAL_EXTRACTOR = "lexical-v1"

Ranking = List[Tuple[str, float]]


class ChannelUnavailable(RuntimeError):
    """A requested stage cannot execute (missing embedder, reranker, extraction).

    Raised instead of silently dropping the stage, so callers never receive a
    ranking produced by fewer signals than they asked for.
    """


@dataclass(frozen=True)
class RetrievalConfig:
    """Every knob that changes a ranking. Hashed into the trace."""

    channels: Tuple[str, ...] = CHANNELS
    top_k: int = 10
    channel_k: int = 100
    fusion: str = "rrf"
    rrf_k: int = 60
    weights: Tuple[Tuple[str, float], ...] = ()
    # dense
    dense_mode: str = "exact"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 40
    hnsw_ef_search: int = 64
    ann_audit: bool = False
    # sparse
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # graph
    graph_method: str = "emg"
    graph_extractor: str = LEXICAL_EXTRACTOR
    graph_query_extractor: str = LEXICAL_EXTRACTOR
    add_speaker_as_entity: bool = True
    sequence_expansion: bool = True
    sequence_scale: float = 0.5
    entity_min_rel_score: float = 0.5
    entity_top_k_per_key: int = 20
    who_only_dampen: float = 0.25
    degree_discount: bool = True
    ppr_damping: float = 0.5
    ppr_linking_top_k: int = 5
    ppr_sequence_edges: bool = True
    ppr_passage_seed_channel: Optional[str] = None
    ppr_passage_node_weight: float = 0.05
    # rerank / evidence
    rerank_pool: int = 0
    evidence_strategy: str = "turn"
    evidence_window: int = 1
    evidence_lambda: float = 0.7
    evidence_budget_tokens: Optional[int] = None
    evidence_order: str = "chronological"

    def weight(self, channel: str) -> float:
        return dict(self.weights).get(channel, 1.0)

    def validate(self) -> None:
        if not self.channels:
            raise ValueError("at least one retrieval channel is required")
        unknown = set(self.channels) - set(CHANNELS)
        if unknown:
            raise ValueError(f"unknown channels: {sorted(unknown)}")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must not repeat")
        if self.top_k < 1 or self.channel_k < 1:
            raise ValueError("top_k and channel_k must be >= 1")
        if self.fusion not in FUSION_MODES:
            raise ValueError(f"fusion must be one of {FUSION_MODES}")
        if not isinstance(self.rrf_k, int) or self.rrf_k < 0:
            raise ValueError("rrf_k must be a non-negative integer")
        for name, value in self.weights:
            if name not in CHANNELS:
                raise ValueError(f"weight given for unknown channel {name!r}")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"weight for {name!r} must be finite and non-negative")
        if self.dense_mode not in DENSE_MODES:
            raise ValueError(f"dense_mode must be one of {DENSE_MODES}")
        if self.dense_mode == "hnsw" and self.fusion in ("zscore", "minmax_linear") and "dense" in self.channels:
            raise ValueError("score-based fusion needs full-scope dense scores; use dense_mode='exact' or rrf/dbsf")
        if self.graph_method not in GRAPH_METHODS:
            raise ValueError(f"graph_method must be one of {GRAPH_METHODS}")
        if self.ppr_passage_seed_channel is not None:
            if self.ppr_passage_seed_channel not in ("dense", "sparse"):
                raise ValueError("ppr_passage_seed_channel must be 'dense', 'sparse' or None")
            if self.graph_method != "ppr":
                raise ValueError("passage seeds only apply to graph_method='ppr'")
            if self.ppr_passage_seed_channel not in self.channels:
                raise ValueError(
                    "ppr passage seeds come from a channel that was not requested; "
                    "request it explicitly so the dependency is visible"
                )
        if not 0.0 < self.ppr_damping < 1.0:
            raise ValueError("ppr_damping must be in (0, 1)")
        if self.rerank_pool < 0:
            raise ValueError("rerank_pool must be >= 0 (0 disables reranking)")
        if 0 < self.rerank_pool < self.top_k:
            raise ValueError("a positive rerank_pool must be >= top_k")
        if self.evidence_strategy not in EVIDENCE_STRATEGIES:
            raise ValueError(f"evidence_strategy must be one of {EVIDENCE_STRATEGIES}")
        if self.evidence_budget_tokens is not None and self.evidence_budget_tokens < 1:
            raise ValueError("evidence_budget_tokens must be positive when set")
        if self.evidence_order not in ("chronological", "rank"):
            raise ValueError("evidence_order must be 'chronological' or 'rank'")

    def resolved(self) -> Dict[str, Any]:
        data = asdict(self)
        data["channels"] = list(self.channels)
        data["weights"] = {c: self.weight(c) for c in self.channels}
        return data


def config_sha256(config: RetrievalConfig) -> str:
    encoded = json.dumps(config.resolved(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ── Channels ──────────────────────────────────────────────────────────────


class SparseChannel:
    """BM25S (Lucene variant) over the scope's index text; scope-local IDF."""

    def __init__(self, corpus: ScopedCorpus, k1: float = 1.5, b: float = 0.75):
        from storage.bm25_index import BM25SBackend

        self.corpus = corpus
        self.identity = f"bm25s-lucene(k1={k1},b={b},stopwords=en,stemmer=snowball)"
        self._backend = BM25SBackend(k1=k1, b=b)
        self._backend.add_batch([(t.node_id, t.search_text) for t in corpus.turns])

    def scores(self, query: str) -> Dict[str, float]:
        """Positive BM25 scores for every matching turn (absent = 0)."""
        return dict(self._backend.search(query, top_k=max(1, len(self.corpus))))

    def rank(self, query: str, k: int) -> Ranking:
        return self.corpus.sort_scores(self.scores(query))[:k]


class DenseChannel:
    """Exact cosine over stored vectors; optional HNSW with measured recall."""

    def __init__(
        self,
        corpus: ScopedCorpus,
        matrix: Any,
        mode: str = "exact",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 40,
        hnsw_ef_search: int = 64,
    ):
        from engine.dense_channel import HNSWScope

        self.corpus = corpus
        self.mode = mode
        self.matrix = matrix
        self._hnsw = (
            HNSWScope(self.matrix, m=hnsw_m, ef_construction=hnsw_ef_construction, ef_search=hnsw_ef_search)
            if mode == "hnsw"
            else None
        )
        self.identity = (
            "dense-exact-ip"
            if mode == "exact"
            else f"dense-hnsw(M={hnsw_m},efC={hnsw_ef_construction},efS={hnsw_ef_search})"
        )

    def exact_scores(self, query_vec: np.ndarray) -> Dict[str, float]:
        from engine.dense_channel import exact_scores

        return exact_scores(query_vec, self.matrix)

    def rank(self, query_vec: np.ndarray, k: int) -> Tuple[Ranking, Dict[str, Any]]:
        from engine.dense_channel import exact_rank

        details: Dict[str, Any] = {"mode": self.mode}
        if self._hnsw is None:
            return exact_rank(query_vec, self.matrix, top_k=k), details
        approx = self._hnsw.rank(query_vec, top_k=k)
        return approx, details

    def audit(self, query_vec: np.ndarray, ranking: Ranking, k: int) -> Dict[str, Any]:
        """Recall@k of the served ranking against exact search for this query."""
        from engine.dense_channel import ann_recall, exact_rank

        exact_ids = [nid for nid, _ in exact_rank(query_vec, self.matrix, top_k=k)]
        served = [nid for nid, _ in ranking[:k]]
        return {"k": k, "recall_vs_exact": ann_recall(exact_ids, served, k)}


class GraphChannel:
    """Entity-memory graph with query-derived anchors (EMG port + PPR)."""

    def __init__(self, corpus: ScopedCorpus, mentions: Mapping[str, Sequence[Any]], extractor_id: str, add_speaker_as_entity: bool):
        from engine.entity_graph import EntityMemoryGraph

        self.corpus = corpus
        self.extractor_id = extractor_id
        self.graph = EntityMemoryGraph.from_corpus(
            corpus, mentions, add_speaker_as_entity=add_speaker_as_entity,
        )
        stats = self.graph.stats()
        self.identity = f"entity-memory-graph(extractor={extractor_id})"
        self.stats = stats

    def rank(
        self,
        query_keys: Sequence[str],
        config: RetrievalConfig,
        passage_scores: Optional[Mapping[str, float]] = None,
    ) -> Ranking:
        if config.graph_method == "emg":
            ranking = self.graph.emg_rank(
                set(query_keys),
                expand_sequence=config.sequence_expansion,
                sequence_secondary_scale=config.sequence_scale,
                min_rel_score=config.entity_min_rel_score,
                top_k_per_key=config.entity_top_k_per_key,
                who_only_dampen=config.who_only_dampen,
                degree_discount=config.degree_discount,
            )
        else:
            ranking = self.graph.ppr_rank(
                set(query_keys),
                damping=config.ppr_damping,
                linking_top_k=config.ppr_linking_top_k,
                include_sequence_edges=config.ppr_sequence_edges,
                passage_scores=passage_scores,
                passage_node_weight=config.ppr_passage_node_weight,
                min_rel_score=config.entity_min_rel_score,
                top_k_per_key=config.entity_top_k_per_key,
            )
        return self.corpus.sort_scores({nid: s for nid, s in ranking if s > 0.0})


# ── Scope index ───────────────────────────────────────────────────────────


class ScopeIndex:
    """Lazily built channel indexes over one ``ScopedCorpus``.

    ``vectors`` returns ``{node_id: vector}`` for the whole scope when the
    dense channel is first used. ``stored_mentions(extractor_id)`` returns
    persisted entity extractions (e.g. from an LLM extractor); lexical
    mentions are derived from turn text and never persisted.
    """

    def __init__(
        self,
        corpus: ScopedCorpus,
        vectors: Optional[Callable[[], Mapping[str, np.ndarray]]] = None,
        expected_dim: Optional[int] = 4096,
        stored_mentions: Optional[Callable[[str], Mapping[str, Sequence[Any]]]] = None,
    ):
        self.corpus = corpus
        self._vectors = vectors
        self.expected_dim = expected_dim
        self._stored_mentions = stored_mentions
        self._sparse: Dict[Tuple[float, float], SparseChannel] = {}
        self._dense: Dict[Tuple[Any, ...], DenseChannel] = {}
        self._graph: Dict[Tuple[str, bool], GraphChannel] = {}
        self._matrix: Any = None

    def sparse(self, config: RetrievalConfig) -> SparseChannel:
        key = (config.bm25_k1, config.bm25_b)
        if key not in self._sparse:
            self._sparse[key] = SparseChannel(self.corpus, k1=config.bm25_k1, b=config.bm25_b)
        return self._sparse[key]

    def dense(self, config: RetrievalConfig) -> DenseChannel:
        if self._vectors is None:
            raise ChannelUnavailable("dense retrieval was requested but this scope has no vector source")
        key = (config.dense_mode, config.hnsw_m, config.hnsw_ef_construction, config.hnsw_ef_search)
        if key not in self._dense:
            if self._matrix is None:
                from engine.dense_channel import DenseMatrix

                # One normalized matrix per scope; the raw vectors are not kept.
                self._matrix = DenseMatrix.from_vectors(self.corpus, self._vectors(), expected_dim=self.expected_dim)
            self._dense[key] = DenseChannel(
                self.corpus,
                self._matrix,
                mode=config.dense_mode,
                hnsw_m=config.hnsw_m,
                hnsw_ef_construction=config.hnsw_ef_construction,
                hnsw_ef_search=config.hnsw_ef_search,
            )
        return self._dense[key]

    def graph(self, config: RetrievalConfig) -> GraphChannel:
        key = (config.graph_extractor, config.add_speaker_as_entity)
        if key not in self._graph:
            mentions = self.mentions(config.graph_extractor)
            self._graph[key] = GraphChannel(
                self.corpus, mentions, config.graph_extractor, config.add_speaker_as_entity,
            )
        return self._graph[key]

    def mentions(self, extractor_id: str) -> Mapping[str, Sequence[Any]]:
        if extractor_id == LEXICAL_EXTRACTOR:
            from engine.entity_extraction import LexicalEntityExtractor

            extractor = LexicalEntityExtractor()
            return {t.node_id: extractor.extract(t.text, speaker=t.speaker) for t in self.corpus.turns}
        if self._stored_mentions is None:
            raise ChannelUnavailable(f"graph extractor {extractor_id!r} needs stored extractions; none are available")
        stored = self._stored_mentions(extractor_id)
        missing = [t.node_id for t in self.corpus.turns if t.node_id not in stored]
        if missing:
            raise ChannelUnavailable(
                f"graph extractor {extractor_id!r} covers {len(self.corpus) - len(missing)}/"
                f"{len(self.corpus)} turns in scope {self.corpus.scope_key!r}; run extraction "
                "for the remaining turns (partial graphs are refused)"
            )
        return stored


# ── Results ───────────────────────────────────────────────────────────────


@dataclass
class ChannelRun:
    name: str
    requested: bool
    executed: bool = False
    candidates: int = 0
    latency_ms: float = 0.0
    identity: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ranking: Ranking = field(default_factory=list)

    def trace(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "executed": self.executed,
            "candidates": self.candidates,
            "latency_ms": self.latency_ms,
            "identity": self.identity,
            **self.details,
        }


@dataclass
class Hit:
    node_id: str
    evidence_id: str
    text: str
    metadata: Dict[str, Any]
    rank: int
    score: float
    channel_ranks: Dict[str, Optional[int]]
    channel_scores: Dict[str, Optional[float]]
    rerank_score: Optional[float] = None

    @property
    def sources(self) -> List[str]:
        return [c for c, r in self.channel_ranks.items() if r is not None]


@dataclass
class RetrievalResult:
    hits: List[Hit]
    fused: Ranking
    channels: Dict[str, ChannelRun]
    evidence: Any
    trace: Dict[str, Any]


# ── Fusion ────────────────────────────────────────────────────────────────


def fuse(corpus: ScopedCorpus, runs: Mapping[str, ChannelRun], config: RetrievalConfig, complete_scores: Mapping[str, Mapping[str, float]]) -> Ranking:
    """Fuse executed channel rankings; single-channel requests pass through."""
    executed = {name: run for name, run in runs.items() if run.executed}
    if len(executed) == 1:
        (only,) = executed.values()
        return list(only.ranking)
    from engine import fusion as F

    weights = {name: config.weight(name) for name in executed}
    if config.fusion == "rrf":
        fused = F.rrf_fuse({n: r.ranking for n, r in executed.items()}, k=config.rrf_k, signal_weights=weights)
    elif config.fusion == "dbsf":
        # Qdrant DBSF normalizes each returned list on its own statistics.
        fused = F.dbsf_fuse({n: dict(r.ranking) for n, r in executed.items()}, weights)
    elif config.fusion == "zscore":
        fused = F.zscore_fuse({n: dict(complete_scores[n]) for n in executed}, weights)
    else:
        normalized = {n: F.minmax_normalize(dict(complete_scores[n])) for n in executed}
        fused = F.weighted_linear_fuse(normalized, weights)
    return corpus.sort_scores(fused)


# ── Retriever ─────────────────────────────────────────────────────────────


class TriSignalRetriever:
    """Runs the requested channels, fuses, reranks and assembles evidence."""

    def __init__(
        self,
        embed_query: Optional[Callable[[str], np.ndarray]] = None,
        reranker: Optional[Any] = None,
        query_key_extractors: Optional[Mapping[str, Callable[[str], Sequence[str]]]] = None,
        embedder_identity: Optional[Mapping[str, Any]] = None,
    ):
        self.embed_query = embed_query
        self.reranker = reranker
        self.embedder_identity = dict(embedder_identity or {})
        self.reranker_identity = None
        if reranker is not None:
            where = getattr(reranker, "base_url", None)
            self.reranker_identity = (
                f"{type(reranker).__name__}({getattr(reranker, 'model_name', '')}"
                + (f"@{where}" if where else "") + ")"
            )
        self.query_key_extractors = dict(query_key_extractors or {})

    def _query_keys(self, question: str, extractor_id: str) -> List[str]:
        if extractor_id in self.query_key_extractors:
            return sorted(set(self.query_key_extractors[extractor_id](question)))
        if extractor_id == LEXICAL_EXTRACTOR:
            from engine.entity_extraction import LexicalEntityExtractor

            return sorted(LexicalEntityExtractor().extract_query(question))
        raise ChannelUnavailable(f"no query-key extractor registered for {extractor_id!r}")

    def retrieve(
        self,
        scope: ScopeIndex,
        query: str,
        config: RetrievalConfig,
        *,
        query_vector: Optional[np.ndarray] = None,
        query_keys: Optional[Sequence[str]] = None,
    ) -> RetrievalResult:
        config.validate()
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        started = time.perf_counter()
        corpus = scope.corpus
        runs = {name: ChannelRun(name=name, requested=name in config.channels) for name in CHANNELS}
        complete: Dict[str, Dict[str, float]] = {}

        if len(corpus) == 0:
            return self._finish([], [], runs, None, scope, config, started, query_keys=None, retriever=self)

        dense_scores: Dict[str, float] = {}
        if "dense" in config.channels:
            t0 = time.perf_counter()
            channel = scope.dense(config)
            if query_vector is None:
                if self.embed_query is None:
                    raise ChannelUnavailable("dense retrieval was requested but no query embedder is configured")
                query_vector = self.embed_query(query)
            ranking, details = channel.rank(query_vector, config.channel_k)
            if config.ann_audit and config.dense_mode == "hnsw":
                details["ann_audit"] = channel.audit(query_vector, ranking, min(config.channel_k, len(corpus)))
            if config.fusion in ("zscore", "minmax_linear") or config.ppr_passage_seed_channel == "dense":
                dense_scores = channel.exact_scores(query_vector)
            self._record(runs["dense"], ranking, channel.identity, details, t0)
            complete["dense"] = dense_scores

        sparse_scores: Dict[str, float] = {}
        if "sparse" in config.channels:
            t0 = time.perf_counter()
            channel = scope.sparse(config)
            sparse_scores = channel.scores(query)
            ranking = corpus.sort_scores(sparse_scores)[: config.channel_k]
            self._record(runs["sparse"], ranking, channel.identity, {}, t0)
            complete["sparse"] = sparse_scores

        keys: Optional[List[str]] = None
        if "graph" in config.channels:
            t0 = time.perf_counter()
            channel = scope.graph(config)
            keys = sorted(set(query_keys)) if query_keys is not None else self._query_keys(query, config.graph_query_extractor)
            seeds = None
            if config.ppr_passage_seed_channel == "dense":
                seeds = dense_scores
            elif config.ppr_passage_seed_channel == "sparse":
                seeds = {t.node_id: sparse_scores.get(t.node_id, 0.0) for t in corpus.turns}
            graph_scores = channel.rank(keys, config, passage_scores=seeds)
            ranking = graph_scores[: config.channel_k]
            details = {
                "method": config.graph_method,
                "query_keys": keys,
                "query_key_source": "caller" if query_keys is not None else config.graph_query_extractor,
                "depends_on": [config.ppr_passage_seed_channel] if seeds is not None else [],
                "graph_stats": channel.stats,
            }
            self._record(runs["graph"], ranking, channel.identity, details, t0)
            complete["graph"] = dict(graph_scores)

        if config.fusion in ("zscore", "minmax_linear"):
            # Score-based fusion normalizes over the whole scope: a turn a
            # channel did not match has that channel's true score of 0.
            for name in ("sparse", "graph"):
                if name in complete:
                    complete[name] = {t.node_id: complete[name].get(t.node_id, 0.0) for t in corpus.turns}
        fused = fuse(corpus, runs, config, complete)
        hits = self._hits(corpus, fused, runs, config)
        hits, rerank_trace = self._rerank(query, hits, config)
        evidence = None
        if config.evidence_budget_tokens is not None or config.evidence_strategy != "turn":
            from engine.evidence import assemble_evidence

            if rerank_trace.get("applied"):
                # After a rerank the evidence order is the rerank order: give
                # every candidate a positive score strictly decreasing by position.
                head = {h.node_id for h in hits}
                order = [h.node_id for h in hits] + [nid for nid, _ in fused if nid not in head]
                ranked_for_evidence = [(nid, float(len(order) - i)) for i, nid in enumerate(order)]
                signed = False
            else:
                ranked_for_evidence = fused
                executed = [n for n, r in runs.items() if r.executed]
                signed = executed == ["dense"] or (len(executed) > 1 and config.fusion in ("zscore", "dbsf"))
            evidence = assemble_evidence(
                corpus,
                ranked_for_evidence,
                signed_scores=signed,
                strategy=config.evidence_strategy,
                window=config.evidence_window,
                lam=config.evidence_lambda,
                budget_tokens=config.evidence_budget_tokens,
                order=config.evidence_order,
            )
        return self._finish(hits[: config.top_k], fused, runs, evidence, scope, config, started, query_keys=keys, rerank=rerank_trace, retriever=self)

    @staticmethod
    def _record(run: ChannelRun, ranking: Ranking, identity: str, details: Dict[str, Any], t0: float) -> None:
        run.executed = True
        run.ranking = list(ranking)
        run.candidates = len(ranking)
        run.identity = identity
        run.details = details
        run.latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    @staticmethod
    def _hits(corpus: ScopedCorpus, fused: Ranking, runs: Mapping[str, ChannelRun], config: RetrievalConfig) -> List[Hit]:
        pool = max(config.top_k, config.rerank_pool)
        ranks = {
            name: {nid: i for i, (nid, _) in enumerate(run.ranking, 1)}
            for name, run in runs.items() if run.executed
        }
        scores = {name: dict(run.ranking) for name, run in runs.items() if run.executed}
        hits = []
        for rank, (nid, score) in enumerate(fused[:pool], 1):
            turn: TurnRecord = corpus.turn(nid)
            hits.append(Hit(
                node_id=nid,
                evidence_id=turn.evidence_id,
                text=turn.text,
                metadata=dict(turn.metadata),
                rank=rank,
                score=float(score),
                channel_ranks={c: ranks[c].get(nid) for c in ranks},
                channel_scores={c: scores[c].get(nid) for c in scores},
            ))
        return hits

    def _rerank(self, query: str, hits: List[Hit], config: RetrievalConfig) -> Tuple[List[Hit], Dict[str, Any]]:
        trace = {"requested": config.rerank_pool > 0, "applied": False, "pool": config.rerank_pool, "identity": None}
        if config.rerank_pool == 0:
            return hits, trace
        if self.reranker is None or not getattr(self.reranker, "enabled", True):
            raise ChannelUnavailable("reranking was requested (rerank_pool > 0) but no reranker is configured")
        pool = hits[: config.rerank_pool]
        candidates = [{"node_id": h.node_id, "text": h.text, "combined_score": h.score} for h in pool]
        reranked = self.reranker.rerank(query, candidates, top_k=None)
        by_id = {h.node_id: h for h in pool}
        failures = sorted({str(c["rerank_failure_type"]) for c in reranked if c.get("rerank_failure_type")})
        if failures or not all("rerank_score" in c for c in reranked):
            raise ChannelUnavailable(f"reranker did not score every pooled candidate ({','.join(failures) or 'missing scores'})")
        # Rerank only reorders the fixed pool; it never admits new candidates.
        ordered = sorted(reranked, key=lambda c: (-float(c["rerank_score"]), by_id[c["node_id"]].rank))
        out = []
        for new_rank, cand in enumerate(ordered, 1):
            hit = by_id[cand["node_id"]]
            hit.rerank_score = float(cand["rerank_score"])
            hit.rank = new_rank
            out.append(hit)
        trace.update(applied=True, identity=self.reranker_identity, candidates=len(out))
        return out + hits[config.rerank_pool:], trace

    @staticmethod
    def _finish(hits, fused, runs, evidence, scope, config, started, query_keys, rerank=None, retriever=None) -> RetrievalResult:
        dense_used = runs["dense"].executed
        execution = {
            "config_sha256": config_sha256(config),
            "as_of": scope.corpus.as_of,
            "embedder": retriever.embedder_identity if (retriever is not None and dense_used) else None,
            "reranker": (rerank or {}).get("identity") if (rerank or {}).get("applied") else None,
        }
        trace = {
            "schema_version": TRACE_SCHEMA,
            "resolved_config": config.resolved(),
            "resolved_config_sha256": config_sha256(config),
            # Everything else that changes a ranking: point in time, query
            # embedder (model, instruction, style) and reranker model.
            "execution": execution,
            "execution_sha256": hashlib.sha256(
                json.dumps(execution, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            ).hexdigest(),
            "scope": {
                "scope_key": scope.corpus.scope_key,
                "generation": scope.corpus.generation,
                "as_of": scope.corpus.as_of,
                "turns": len(scope.corpus),
            },
            "channels": {name: run.trace() for name, run in runs.items()},
            "fusion": {
                "mode": config.fusion if sum(r.executed for r in runs.values()) > 1 else "passthrough",
                "rrf_k": config.rrf_k if config.fusion == "rrf" else None,
                "candidates": len(fused),
            },
            "rerank": rerank or {"requested": False, "applied": False, "pool": 0, "identity": None},
            "evidence": None if evidence is None else {
                "strategy": evidence.strategy,
                "budget_tokens": evidence.budget_tokens,
                "packed_tokens": evidence.packed_tokens,
                "items": len(evidence.items),
                "dropped_hits": evidence.dropped_hits,
            },
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        return RetrievalResult(hits=hits, fused=fused, channels=dict(runs), evidence=evidence, trace=trace)


# ── SQLite-backed scope registry (service path) ───────────────────────────


def canonical_scope(filters: Optional[Mapping[str, Any]]) -> str:
    return json.dumps(dict(filters or {}), sort_keys=True, separators=(",", ":"), default=str)


class ScopeRegistry:
    """Builds and caches ``ScopeIndex`` objects from the authoritative store.

    The cache key includes the store's ``corpus_generation`` so any mutation
    (node, edge, entity or extraction write) invalidates derived indexes.
    """

    def __init__(self, sqlite_store: Any, max_scopes: int = 32):
        self.store = sqlite_store
        self.max_scopes = max_scopes
        self._cache: "OrderedDict[Tuple[str, int, Optional[str]], ScopeIndex]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, filters: Optional[Mapping[str, Any]], as_of: Optional[str] = None) -> ScopeIndex:
        from engine.corpus import turn_from_node

        generation = self.store.get_corpus_generation()
        key = (canonical_scope(filters), generation, as_of)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            # Any write bumps the generation; indexes of older generations can
            # never be served again, so free them now rather than by LRU.
            for stale in [k for k in self._cache if k[1] != generation]:
                del self._cache[stale]
        nodes = self.store.list_scope_nodes(filters or {}, as_of=as_of)
        corpus = ScopedCorpus(
            (turn_from_node(n) for n in nodes), scope_key=key[0], generation=generation, as_of=as_of,
        )
        ids = [t.node_id for t in corpus.turns]
        index = ScopeIndex(
            corpus,
            vectors=lambda: self._scope_vectors(ids),
            expected_dim=4096,
            stored_mentions=lambda extractor: {
                nid: [_mention(m) for m in ms]
                for nid, ms in self.store.get_entity_extractions(ids, extractor).items()
            },
        )
        with self._lock:
            self._cache[key] = index
            while len(self._cache) > self.max_scopes:
                self._cache.popitem(last=False)
        return index

    def _scope_vectors(self, ids: List[str]) -> Mapping[str, np.ndarray]:
        vectors = self.store.get_node_embeddings(ids)
        missing = len(ids) - len(vectors)
        if missing:
            raise ChannelUnavailable(
                f"dense retrieval needs a stored embedding for every turn in scope; {missing}/{len(ids)} have none"
            )
        return vectors


def _mention(payload: Mapping[str, Any]) -> Any:
    from engine.entity_extraction import EntityMention

    return EntityMention(key=payload["key"], value=payload.get("value", payload["key"]), type=payload.get("type", ""))
