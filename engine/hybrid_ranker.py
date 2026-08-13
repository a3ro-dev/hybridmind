"""
Hybrid ranker for HybridMind.
Implements the Contextual Relevance Score (CRS) algorithm with:
  - 4-signal RRF fusion: dense + sparse (BM25S/SPLADE) + graph (temporal) + ColBERT MaxSim
  - Query-type-aware weight routing (temporal/multihop/entity/factual)
  - Temporal graph decay (optional, HYBRIDMIND_TEMPORAL_DECAY_ENABLED=true)
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from engine.vector_search import VectorSearchEngine
from engine.graph_search import GraphSearchEngine
from engine.fusion import rrf_fuse, get_fusion_mode, get_rrf_k
from engine.query_router import route_query
from engine.salience import compute_salience
from engine.temporal import extract_time_range, parse_datetime, temporal_relevance, validity_relevance
from models.edge import EDGE_TYPE_WALK_WEIGHTS

logger = logging.getLogger(__name__)

_SEARCH_MODES = {"hybrid", "vector_only", "sparse_only", "vector_sparse", "graph_only"}


def _query_type_weights(query_type: str) -> Dict[str, float]:
    from config import settings
    prefix = query_type if query_type in {"temporal", "multihop", "entity"} else "default"
    return {
        "vector_weight": getattr(settings, f"route_{prefix}_vector_weight"),
        "graph_weight": getattr(settings, f"route_{prefix}_graph_weight"),
        "bm25_boost_weight": getattr(settings, f"route_{prefix}_sparse_weight"),
        "time_weight": getattr(settings, "route_temporal_time_weight", 0.0) if prefix == "temporal" else 0.0,
    }


class HybridRanker:
    """
    Hybrid search ranker combining vector similarity and graph proximity.

    Implements the Contextual Relevance Score (CRS) algorithm:
    Score = α * V(q, n) + β * G(A, n)

    Where:
    - α (vector_weight): Weight for semantic similarity
    - β (graph_weight): Weight for graph proximity
    """

    def __init__(
        self,
        vector_engine: VectorSearchEngine,
        graph_engine: GraphSearchEngine,
        bm25_index: Optional[Any] = None,
        disable_graph_expansion: bool = False,
        reranker: Optional[Any] = None,
        query_routing_enabled: Optional[bool] = None,
        temporal_decay_enabled: Optional[bool] = None,
        temporal_decay_half_life_days: float = 30.0,
    ):
        self.vector_engine = vector_engine
        self.graph_engine = graph_engine
        self.bm25_index = bm25_index
        self.disable_graph_expansion = disable_graph_expansion
        self.reranker = reranker
        self._lexical_term_cache = None
        if bm25_index is not None:
            try:
                from config import settings as _cfg
                if _cfg.local_lexical_term_cache_size > 0:
                    from engine.lexical_reranker import BoundedTokenSetCache
                    self._lexical_term_cache = BoundedTokenSetCache(
                        bm25_index.tokenize,
                        max_entries=_cfg.local_lexical_term_cache_size,
                    )
            except Exception as e:
                logger.debug("Query-local lexical token cache disabled (%s)", type(e).__name__)

        # Query routing: classify query → per-type weights
        # Reads from config if not explicitly set
        if query_routing_enabled is None:
            try:
                from config import settings as _cfg
                query_routing_enabled = _cfg.query_routing_enabled
            except Exception:
                query_routing_enabled = True
        self.query_routing_enabled = query_routing_enabled

        # Temporal decay: weight graph edges by recency
        if temporal_decay_enabled is None:
            try:
                from config import settings as _cfg
                temporal_decay_enabled = _cfg.temporal_decay_enabled
                temporal_decay_half_life_days = _cfg.temporal_decay_half_life_days
            except Exception:
                temporal_decay_enabled = False
        self.temporal_decay_enabled = temporal_decay_enabled
        self.temporal_decay_half_life_days = temporal_decay_half_life_days

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        vector_weight: Optional[float] = None,
        graph_weight: Optional[float] = None,
        anchor_nodes: Optional[List[str]] = None,
        max_depth: int = 2,
        min_score: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
        deduplicate: bool = True,
        search_mode: str = "hybrid",
        bm25_boost_weight: Optional[float] = None,
        rerank_pool: int = 25,
        overlap_threshold: float = 0.15,
        fusion_mode: Optional[str] = None,
        include_images: bool = False,
        route_weights: bool = True,
        track_access: Optional[bool] = None,
    ) -> Tuple[List[Dict[str, Any]], float, int]:
        start_time = time.perf_counter()
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if rerank_pool < 0:
            raise ValueError("rerank_pool must be non-negative; use 0 to disable reranking")
        if 0 < rerank_pool < top_k:
            raise ValueError("positive rerank_pool must be greater than or equal to top_k")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        if search_mode not in _SEARCH_MODES:
            raise ValueError(f"Unsupported search_mode {search_mode!r}; expected one of {sorted(_SEARCH_MODES)}")
        if search_mode == "graph_only" and not anchor_nodes:
            raise ValueError("graph_only search requires at least one anchor node")

        use_dense = search_mode in {"hybrid", "vector_only", "vector_sparse"}
        use_sparse = search_mode in {"hybrid", "sparse_only", "vector_sparse"}
        use_graph = search_mode in {"hybrid", "graph_only"}

        # ``None`` means the caller omitted a weight and permits routing to
        # supply it. Explicit values always win, even when routing is enabled.
        explicit_weights = {
            "vector": vector_weight is not None,
            "graph": graph_weight is not None,
            "sparse": bm25_boost_weight is not None,
        }
        defaults = _query_type_weights("default")
        vector_weight = defaults["vector_weight"] if vector_weight is None else vector_weight
        graph_weight = defaults["graph_weight"] if graph_weight is None else graph_weight
        bm25_boost_weight = (
            defaults["bm25_boost_weight"]
            if bm25_boost_weight is None
            else bm25_boost_weight
        )
        for name, value in (
            ("vector_weight", vector_weight),
            ("graph_weight", graph_weight),
            ("bm25_boost_weight", bm25_boost_weight),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        # ── Query-type-aware weight routing ────────────────────────────────
        # Classify the query and route only weights the caller omitted.
        # Every explicit caller value is preserved, including a value equal to
        # the default.
        query_type = "default"
        temporal_order_intent: Optional[str] = None
        time_weight = 0.0
        route = {"type": "default", "metadata_filter": None}
        if self.query_routing_enabled:
            try:
                route = route_query(query_text)
                query_type = route.get("type", "default")
                if re.search(r"\b(latest|most recent|current)\b", query_text, re.IGNORECASE):
                    query_type = "temporal"
                    temporal_order_intent = "latest"
                elif re.search(r"\b(previous|prior)\b", query_text, re.IGNORECASE):
                    query_type = "temporal"
                    temporal_order_intent = "previous"
                overrides = _query_type_weights(query_type)
                if route_weights:
                    if not explicit_weights["vector"]:
                        vector_weight = overrides["vector_weight"]
                    if not explicit_weights["graph"]:
                        graph_weight = overrides["graph_weight"]
                    if not explicit_weights["sparse"]:
                        bm25_boost_weight = overrides["bm25_boost_weight"]
                    time_weight = overrides["time_weight"]
            except Exception as _e:
                logger.debug("Query routing failed (using defaults): %s", type(_e).__name__)
        if not use_dense:
            vector_weight = 0.0
        if not use_sparse:
            bm25_boost_weight = 0.0
        if not use_graph:
            graph_weight = 0.0
        if query_type != "temporal" or search_mode != "hybrid":
            time_weight = 0.0

        route_filter = route.get("metadata_filter") or {}
        effective_filter = {**route_filter, **(filter_metadata or {})} or None

        # We need candidate generation. We will pull top_k * 5 vector results and bm25 results.
        vector_k = top_k * 5 if deduplicate else top_k * 3
        # Expand candidates to accommodate SGMem sentence chunks
        vector_k = max(40, vector_k * 2)

        # We also need a larger candidate pool if graph helps recall
        candidate_k = max(100, vector_k)

        # Step 1: Run Vector and BM25 search (with sub-query decomposition for multi-hop queries)
        sub_questions = []
        if query_type == "multihop" and search_mode == "hybrid":
            try:
                from config import settings as _cfg
                from engine.query_decomposition import decompose_query
                sub_questions = decompose_query(
                    query_text,
                    model=_cfg.query_decomposition_model,
                    enabled=_cfg.query_decomposition_enabled,
                )
            except Exception as exc:
                logger.debug("Query decomposition skipped (%s)", type(exc).__name__)

        queries_to_search = [query_text] + sub_questions
        vector_results = []
        seen_vec_ids = set()

        if use_dense:
            for q in queries_to_search:
                v_res, _, _ = self.vector_engine.search(
                    query_text=q,
                    top_k=candidate_k,
                    min_score=0.0,
                    filter_metadata=effective_filter
                )
                for r in v_res:
                    if r["node_id"] not in seen_vec_ids:
                        seen_vec_ids.add(r["node_id"])
                        vector_results.append(r)

        bm25_results = []
        bm25_result_ids: Set[str] = set()
        bm25_score_by_node: Dict[str, float] = {}
        if self.bm25_index and use_sparse:
            for q in queries_to_search:
                bm25_hits = self.bm25_index.search(q, top_k=5000)
                for n_id, score in bm25_hits:
                    if not self.vector_engine.sqlite_store.is_node_retrievable(n_id):
                        continue
                    if n_id not in bm25_score_by_node or score > bm25_score_by_node[n_id]:
                        bm25_score_by_node[n_id] = score
                    node = self.vector_engine.sqlite_store.get_node(n_id)
                    if node:
                        if effective_filter and not self.vector_engine._matches_filter(node["metadata"], effective_filter):
                            continue
                        if n_id not in bm25_result_ids:
                            bm25_result_ids.add(n_id)
                            bm25_results.append({
                                "node_id": n_id,
                                "text": node["text"],
                                "metadata": node["metadata"],
                                "bm25_score": score
                            })
                        if len(bm25_results) >= candidate_k * 3:
                            break

        # Step 2: Combine vector and BM25 into a baseline V score.
        # Vector score is cosine similarity (0 to 1). We should boost it if BM25 matches.
        def bm25_overlap(query: str, text: str) -> float:
            if not self.bm25_index or not use_sparse:
                return 0.0
            q_terms = set(self.bm25_index.tokenize(query))
            t_terms = set(self.bm25_index.tokenize(text))
            if not q_terms: return 0.0
            overlap = sum(1 for qt in q_terms if qt in t_terms)
            return overlap / len(q_terms)

        scores = {}
        raw_vector_scores = {}  # Track raw cosine similarity (pre-BM25 boost) for relevance gating
        raw_bm25_scores = {}    # Real BM25 relevance scores — used as the dedicated "sparse" RRF signal
        node_data = {}

        # Give vector results their base cosine score + bm25 overlap boost
        for res in vector_results:
            nid = res["node_id"]
            node_data[nid] = res
            raw_v = res.get("vector_score", 0.0)
            raw_vector_scores[nid] = raw_v
            raw_bm25_scores[nid] = bm25_score_by_node.get(nid, 0.0)
            boost = bm25_overlap(query_text, res["text"]) * bm25_boost_weight
            scores[nid] = raw_v + boost

        # Give BM25 results a baseline score if they did not have a vector score.
        # Sparse-only is an exact signal control: both its candidate pool and
        # output order are driven by the raw BM25 score, never token-overlap or
        # a fabricated vector score.
        for res in bm25_results:
            nid = res["node_id"]
            if nid not in node_data:
                node_data[nid] = res
                raw_bm25 = res.get("bm25_score", bm25_score_by_node.get(nid, 0.0))
                if search_mode == "sparse_only":
                    synthetic_base = 0.0
                    boost = 0.0
                    baseline_score = raw_bm25
                else:
                    overlap = bm25_overlap(query_text, res["text"])
                    boost = overlap * bm25_boost_weight
                    synthetic_base = min(overlap * 1.5, 1.0) if overlap > 0 else 0.0
                    baseline_score = synthetic_base + boost
                node_data[nid]["vector_score"] = synthetic_base
                raw_vector_scores[nid] = 0.0  # no real vector/cosine score for BM25-only hits
                raw_bm25_scores[nid] = raw_bm25
                scores[nid] = baseline_score

        # Graph-only candidates are generated strictly from explicit anchors;
        # no hidden dense seed is introduced into this ablation condition.
        if search_mode == "graph_only":
            for anchor in anchor_nodes or []:
                anchor_node = self.vector_engine.sqlite_store.get_node(anchor)
                if anchor_node is None:
                    raise ValueError(f"graph_only anchor does not exist: {anchor}")
                if effective_filter and not self.vector_engine._matches_filter(
                    anchor_node["metadata"], effective_filter
                ):
                    raise ValueError(
                        "graph_only anchor is outside the requested metadata scope"
                    )
                node_data[anchor] = {
                    "node_id": anchor,
                    "text": anchor_node["text"],
                    "metadata": anchor_node["metadata"],
                    "vector_score": 0.0,
                }
                scores[anchor] = 0.0
                raw_vector_scores[anchor] = 0.0
                raw_bm25_scores[anchor] = 0.0
                traversed, _, _ = self.graph_engine.traverse(
                    anchor, depth=max_depth, direction="typed"
                )
                for item in traversed:
                    if effective_filter and not self.vector_engine._matches_filter(
                        item["metadata"], effective_filter
                    ):
                        continue
                    nid = item["node_id"]
                    node_data[nid] = {
                        "node_id": nid,
                        "text": item["text"],
                        "metadata": item["metadata"],
                        "vector_score": 0.0,
                    }
                    scores[nid] = 0.0
                    raw_vector_scores[nid] = 0.0
                    raw_bm25_scores[nid] = 0.0

        # Step 3: SGMem Chunk Rollup to Parent
        rolled_up_scores = {}
        rolled_up_raw_scores = {}  # Raw vector scores after rollup (for relevance gating)
        rolled_up_raw_bm25_scores = {}  # Real BM25 scores after rollup (RRF "sparse" signal)
        rolled_up_nodes = {}

        for nid, score in scores.items():
            raw_v = raw_vector_scores.get(nid, 0.0)
            raw_bm25 = raw_bm25_scores.get(nid, 0.0)
            meta = node_data[nid].get("metadata", {})
            if meta.get("is_sentence_chunk") and meta.get("parent_id"):
                parent_id = meta["parent_id"]
                rolled_up_scores[parent_id] = max(rolled_up_scores.get(parent_id, 0.0), score)
                rolled_up_raw_scores[parent_id] = max(rolled_up_raw_scores.get(parent_id, 0.0), raw_v)
                rolled_up_raw_bm25_scores[parent_id] = max(rolled_up_raw_bm25_scores.get(parent_id, 0.0), raw_bm25)
                if parent_id not in rolled_up_nodes:
                    p_node = self.vector_engine.sqlite_store.get_node(parent_id)
                    if p_node:
                        p_data = {
                            "node_id": parent_id,
                            "text": p_node["text"],
                            "metadata": p_node["metadata"],
                            "vector_score": node_data[nid].get("vector_score", 0.0)
                        }
                        rolled_up_nodes[parent_id] = p_data
            else:
                rolled_up_scores[nid] = score
                rolled_up_raw_scores[nid] = raw_v
                rolled_up_raw_bm25_scores[nid] = raw_bm25
                rolled_up_nodes[nid] = node_data[nid]

        # Hybrid/vector baseline scores are API-normalized. Sparse-only keeps
        # the native BM25 values intact so truncation cannot turn large scores
        # into ties or re-order its candidate pool.
        if search_mode != "sparse_only":
            for nid in rolled_up_scores:
                rolled_up_scores[nid] = min(rolled_up_scores[nid], 1.0)

        candidate_signal_scores = (
            rolled_up_raw_bm25_scores
            if search_mode == "sparse_only"
            else rolled_up_scores
        )
        sorted_rrf = sorted(candidate_signal_scores.items(), key=lambda x: -x[1])
        candidate_ids = [nid for nid, _ in sorted_rrf[:candidate_k] if nid in rolled_up_nodes]

        if not candidate_ids:
            return [], round((time.perf_counter() - start_time) * 1000, 2), 0

        # Deduplicate candidate_ids by text: keep only the highest-scoring node_id for each unique text.
        # This prevents the same text (from different containers) from flooding the candidate pool.
        if deduplicate:
            seen_texts: set = set()
            deduped_ids = []
            for nid in candidate_ids:
                nd = rolled_up_nodes.get(nid)
                if nd is None:
                    continue
                tk = nd.get("text", "").strip()
                if tk in seen_texts:
                    continue
                seen_texts.add(tk)
                deduped_ids.append(nid)
            candidate_ids = deduped_ids

        # Optional graph-aware candidate expansion path:
        # Before we compute graph scores, add graph neighbors of anchor nodes to candidate pool
        # This allows graph structure to affect recall
        if anchor_nodes and use_graph and not self.disable_graph_expansion:
            reference_nodes = anchor_nodes
        else:
            reference_nodes = candidate_ids[:3]

        # Expand candidates without losing rank order. A set here made tied
        # zero-score candidates depend on Python hash iteration order.
        expanded_candidates = list(candidate_ids)
        expanded_candidate_ids = set(candidate_ids)
        if use_graph and not self.disable_graph_expansion:
            for ref in reference_nodes:
                # Traversal
                try:
                    # get nodes from graph within max_depth
                    neighbors, _, _ = self.graph_engine.traverse(
                        start_id=ref, depth=max_depth, direction="typed"
                    )
                    for n in neighbors:
                        neighbor_id = n["node_id"]
                        if not self.vector_engine.sqlite_store.is_node_retrievable(neighbor_id):
                            continue
                        if neighbor_id not in expanded_candidate_ids:
                            expanded_candidate_ids.add(neighbor_id)
                            expanded_candidates.append(neighbor_id)
                except Exception as e:
                    logger.warning(
                        "Graph traversal failed for an anchor (%s)", type(e).__name__
                    )

        # Add the expanded candidates to our node_data and rolled_up_scores if missing
        for nid in expanded_candidates:
            if nid not in rolled_up_nodes:
                p_node = self.vector_engine.sqlite_store.get_node(nid)
                if p_node:
                    if effective_filter and not self.vector_engine._matches_filter(p_node["metadata"], effective_filter):
                        continue
                    p_data = {
                        "node_id": nid,
                        "text": p_node["text"],
                        "metadata": p_node["metadata"],
                        "vector_score": 0.0 # pure graph candidates have 0 initial vector score
                    }
                    rolled_up_nodes[nid] = p_data
                    rolled_up_scores[nid] = 0.0
                    rolled_up_raw_scores[nid] = 0.0
                    rolled_up_raw_bm25_scores[nid] = 0.0

        # Update candidate_ids to include expanded pool
        candidate_ids = [nid for nid in expanded_candidates if nid in rolled_up_nodes]

        # Step 4: Compute Graph Scores (with optional temporal decay)
        graph_scores = (
            self.graph_engine.compute_proximity_scores(
                node_ids=candidate_ids,
                reference_nodes=reference_nodes,
                max_depth=max_depth,
                edge_type_weights=EDGE_TYPE_WALK_WEIGHTS,
                # Edge recency is a lifecycle/temporal policy, not part of a
                # pure graph signal control.
                temporal_decay=self.temporal_decay_enabled and search_mode == "hybrid",
                half_life_days=self.temporal_decay_half_life_days,
            )
            if use_graph
            else {nid: 0.0 for nid in candidate_ids}
        )

        from config import settings as _cfg
        lifecycle_active = search_mode == "hybrid"
        target_time = (
            extract_time_range(query_text)
            if lifecycle_active and _cfg.query_time_expansion_enabled
            else None
        )
        time_scores: Dict[str, float] = {}
        salience_scores: Dict[str, float] = {}
        validity_scores: Dict[str, float] = {}
        salience_active = _cfg.salience_enabled and lifecycle_active
        graph_max_degree = None
        if salience_active:
            graph = self.graph_engine.graph_index.graph
            graph_max_degree = max((degree for _, degree in graph.degree()), default=1)
        for nid in candidate_ids:
            node = self.vector_engine.sqlite_store.get_node(nid)
            time_scores[nid] = (
                temporal_relevance(
                    (node or {}).get("event_time") or rolled_up_nodes[nid]["metadata"].get("date"),
                    target_time,
                    half_life_days=self.temporal_decay_half_life_days,
                )
                if lifecycle_active
                else 0.0
            )
            salience_scores[nid] = (
                compute_salience(
                    node,
                    self.graph_engine.graph_index,
                    _cfg,
                    max_degree=graph_max_degree,
                )
                if salience_active and node is not None
                else 0.0
            )
            validity_scores[nid] = (
                validity_relevance(node or {}, target_time)
                if lifecycle_active
                else 1.0
            )

        if lifecycle_active and temporal_order_intent:
            dated = []
            for nid in candidate_ids:
                node = self.vector_engine.sqlite_store.get_node(nid) or {}
                timestamp = parse_datetime(
                    node.get("event_time")
                    or rolled_up_nodes[nid]["metadata"].get("date")
                    or rolled_up_nodes[nid]["metadata"].get("timestamp")
                )
                if timestamp is not None:
                    dated.append((nid, timestamp))
            ordered_times = sorted({timestamp for _, timestamp in dated}, reverse=True)
            time_rank = {timestamp: rank for rank, timestamp in enumerate(ordered_times, 1)}
            desired_rank = 1 if temporal_order_intent == "latest" else 2
            for nid, timestamp in dated:
                time_scores[nid] = 1.0 if time_rank[timestamp] == desired_rank else 0.0

        # Step 5: Late Fusion Scoring
        fusion_mode = fusion_mode or get_fusion_mode()

        if search_mode == "sparse_only":
            max_bm25 = max(
                (rolled_up_raw_bm25_scores.get(nid, 0.0) for nid in candidate_ids),
                default=0.0,
            )
            hybrid_results = []
            for nid in candidate_ids:
                b_score = rolled_up_raw_bm25_scores.get(nid, 0.0)
                normalized_bm25 = b_score / max_bm25 if max_bm25 > 0.0 else 0.0
                hybrid_results.append({
                    "node_id": nid,
                    "text": rolled_up_nodes[nid]["text"],
                    "metadata": rolled_up_nodes[nid]["metadata"],
                    "vector_score": 0.0,
                    "raw_vector_score": 0.0,
                    "bm25_score": b_score,
                    "time_score": 0.0,
                    "salience_score": 0.0,
                    "graph_score": 0.0,
                    "graph_gate": 1.0,
                    "effective_graph_score": 0.0,
                    "combined_score": min(1.0, max(0.0, normalized_bm25)),
                    "reasoning": f"raw_bm25={b_score:.4f}",
                    "fusion_mode": "raw_bm25",
                    "query_type": query_type,
                })
        elif fusion_mode == "rrf":
            # Build per-signal rank lists (sorted descending by score).
            # "dense" uses the true cosine score (rolled_up_raw_scores), NOT the
            # BM25-overlap-boosted `rolled_up_scores` — BM25 relevance now gets its
            # own dedicated "sparse" RRF signal below instead of being pre-baked
            # into the dense signal, so it isn't double-counted.
            dense_list = sorted(
                [
                    (nid, rolled_up_raw_scores.get(nid, 0.0))
                    for nid in candidate_ids
                    if rolled_up_raw_scores.get(nid, 0.0) > 0.0
                ],
                key=lambda x: -x[1],
            )
            sparse_list = sorted(
                [
                    (nid, rolled_up_raw_bm25_scores.get(nid, 0.0))
                    for nid in candidate_ids
                    if rolled_up_raw_bm25_scores.get(nid, 0.0) > 0.0
                ],
                key=lambda x: -x[1],
            )
            graph_list = sorted(
                [
                    (nid, graph_scores.get(nid, 0.0))
                    for nid in candidate_ids
                    if graph_scores.get(nid, 0.0) > 0.0
                ],
                key=lambda x: -x[1],
            )
            time_list = sorted(
                [(nid, time_scores[nid]) for nid in candidate_ids if time_scores[nid] > 0.0],
                key=lambda x: -x[1],
            )
            rrf_k = get_rrf_k()
            signal_weights = {
                "dense": vector_weight,
                "graph": graph_weight,
                "sparse": bm25_boost_weight,
                "time": time_weight,
            }
            rrf_scores = rrf_fuse(
                {"dense": dense_list, "graph": graph_list, "sparse": sparse_list, "time": time_list},
                k=rrf_k,
                signal_weights=signal_weights,
            )
            # Convert RRF's natural ~1/(k+rank) scale to the API's documented
            # [0, 1] threshold scale using the theoretical rank-1 ceiling.
            rrf_ceiling = sum(max(0.0, float(w)) for w in signal_weights.values()) / (rrf_k + 1)
            hybrid_results = []
            for nid in candidate_ids:
                v_score = rolled_up_raw_scores.get(nid, 0.0)
                g_score = graph_scores.get(nid, 0.0)
                b_score = rolled_up_raw_bm25_scores.get(nid, 0.0)
                salience = salience_scores.get(nid, 0.0)
                normalized_rrf = rrf_scores.get(nid, 0.0) / max(rrf_ceiling, 1e-12)
                combined_score = min(1.0, normalized_rrf * validity_scores[nid] * (
                    1.0 + (_cfg.salience_weight * salience if salience_active else 0.0)
                ))
                hybrid_results.append({
                    "node_id": nid,
                    "text": rolled_up_nodes[nid]["text"],
                    "metadata": rolled_up_nodes[nid]["metadata"],
                    # Display vector_score as the boosted/back-compat value so existing
                    # consumers (linear-mode UI, eval scripts) see the same field shape.
                    "vector_score": (
                        v_score
                        if search_mode == "sparse_only"
                        else rolled_up_scores.get(nid, v_score)
                    ),
                    "raw_vector_score": v_score,
                    "bm25_score": b_score,
                    "time_score": time_scores.get(nid, 0.0),
                    "salience_score": salience,
                    "graph_score": g_score,
                    "graph_gate": 1.0,
                    "effective_graph_score": round(g_score, 4),
                    "combined_score": combined_score,
                    "reasoning": f"RRF(dense={v_score:.4f}, sparse={b_score:.4f}, graph={g_score:.4f}, time={time_scores.get(nid, 0.0):.4f}, salience={salience:.4f}, qtype={query_type})",
                    "fusion_mode": "rrf",
                    "query_type": query_type,
                })
        elif fusion_mode == "linear":
            # Linear fusion (original CRS algorithm) with relevance gate — kept for back-compat
            hybrid_results = []
            max_bm25 = max(
                (rolled_up_raw_bm25_scores.get(nid, 0.0) for nid in candidate_ids),
                default=0.0,
            )
            for nid in candidate_ids:
                v_score = (
                    rolled_up_raw_scores.get(nid, 0.0)
                    if search_mode == "sparse_only"
                    else rolled_up_scores[nid]
                )
                g_score = graph_scores.get(nid, 0.0)

                # BM25 overlap gate: graph score should amplify already-relevant nodes
                overlap = bm25_overlap(query_text, rolled_up_nodes[nid]["text"])
                if g_score > 0.0:
                    gate = 1.0
                else:
                    gate = min(1.0, overlap / overlap_threshold) if overlap_threshold > 0.0 else 1.0
                effective_g_score = g_score * gate

                linear_weight_sum = max(
                    vector_weight + bm25_boost_weight + graph_weight + time_weight,
                    1e-12,
                )
                combined_score = min(1.0, (
                    (vector_weight * v_score)
                    + (
                        bm25_boost_weight
                        * (
                            rolled_up_raw_bm25_scores.get(nid, 0.0) / max_bm25
                            if max_bm25 > 0.0
                            else 0.0
                        )
                    )
                    + (graph_weight * effective_g_score)
                    + (time_weight * time_scores.get(nid, 0.0))
                ) / linear_weight_sum * validity_scores[nid])
                hybrid_results.append({
                    "node_id": nid,
                    "text": rolled_up_nodes[nid]["text"],
                    "metadata": rolled_up_nodes[nid]["metadata"],
                    "vector_score": v_score,
                    "graph_score": g_score,
                    "bm25_score": rolled_up_raw_bm25_scores.get(nid, 0.0),
                    "time_score": time_scores.get(nid, 0.0),
                    "salience_score": salience_scores.get(nid, 0.0),
                    "graph_gate": round(gate, 4),
                    "effective_graph_score": round(effective_g_score, 4),
                    "combined_score": combined_score,
                    "reasoning": f"Score = {vector_weight}*{v_score:.4f} + {graph_weight}*{effective_g_score:.4f} (gate={gate:.2f})",
                    "fusion_mode": "linear",
                })
        elif fusion_mode == "mlp":
            from engine.fusion import _build_feature_vector, get_fusion_scorer

            dense_rank = {nid: rank for rank, (nid, _) in enumerate(sorted(
                ((nid, rolled_up_raw_scores.get(nid, 0.0)) for nid in candidate_ids),
                key=lambda item: -item[1],
            ), 1)}
            sparse_rank = {nid: rank for rank, (nid, _) in enumerate(sorted(
                ((nid, rolled_up_raw_bm25_scores.get(nid, 0.0)) for nid in candidate_ids),
                key=lambda item: -item[1],
            ), 1)}
            graph_rank = {nid: rank for rank, (nid, _) in enumerate(sorted(
                graph_scores.items(), key=lambda item: -item[1]
            ), 1)}
            scorer = get_fusion_scorer()
            hybrid_results = []
            for nid in candidate_ids:
                features = _build_feature_vector(
                    rolled_up_raw_scores.get(nid, 0.0),
                    rolled_up_raw_bm25_scores.get(nid, 0.0),
                    graph_scores.get(nid, 0.0),
                    dense_rank[nid], sparse_rank[nid], graph_rank[nid],
                    len(candidate_ids), query_type=query_type,
                )
                score = scorer.score(features) * validity_scores[nid]
                hybrid_results.append({
                    "node_id": nid,
                    "text": rolled_up_nodes[nid]["text"],
                    "metadata": rolled_up_nodes[nid]["metadata"],
                    "vector_score": (
                        rolled_up_raw_scores.get(nid, 0.0)
                        if search_mode == "sparse_only"
                        else rolled_up_scores.get(nid, 0.0)
                    ),
                    "bm25_score": rolled_up_raw_bm25_scores.get(nid, 0.0),
                    "graph_score": graph_scores.get(nid, 0.0),
                    "time_score": time_scores.get(nid, 0.0),
                    "salience_score": salience_scores.get(nid, 0.0),
                    "graph_gate": 1.0,
                    "effective_graph_score": graph_scores.get(nid, 0.0),
                    "combined_score": score,
                    "reasoning": f"MLP fusion(qtype={query_type})",
                    "fusion_mode": "mlp",
                    "query_type": query_type,
                })
        else:
            raise ValueError("fusion_mode must be one of: rrf, linear, mlp")

        if deduplicate:
            seen_texts: Set[str] = set()
            deduped = []
            dedup_score_field = {
                "vector_only": "raw_vector_score",
                "sparse_only": "bm25_score",
                "graph_only": "graph_score",
                "vector_sparse": "combined_score",
            }.get(search_mode, "vector_score")
            hybrid_results.sort(key=lambda x: -x.get(dedup_score_field, 0.0))
            for result in hybrid_results:
                text_key = result["text"].strip()
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    deduped.append(result)
            hybrid_results = deduped

        hybrid_results.sort(key=lambda x: (-x["combined_score"], -x.get("vector_score", 0.0)))
        # Hybrid enforces fact validity before downstream rerankers. Controlled
        # signal modes intentionally bypass that lifecycle policy.
        hybrid_results = [
            r for r in hybrid_results
            if (not lifecycle_active or validity_scores.get(r["node_id"], 1.0) > 0.0)
            and r["combined_score"] >= min_score
        ]

        # Controlled modes are signal ablations, not text-reranker ablations.
        # Keep them pure by bypassing every downstream learned/lexical stage.
        apply_post_rerankers = search_mode == "hybrid"

        # ── ColBERT MaxSim late interaction (opt-in: HYBRIDMIND_COLBERT_ENABLED=true) ──
        try:
            from storage.colbert_store import colbert_enabled
            if apply_post_rerankers and colbert_enabled():
                from engine.colbert_reranker import colbert_maxsim_rerank
                # The embedding engine is accessible via the vector_engine
                emb_engine = self.vector_engine.embedding_engine
                hybrid_results = colbert_maxsim_rerank(
                    query_text, hybrid_results, emb_engine,
                )
        except ImportError:
            pass

        # ── GNN reranker (opt-in: HYBRIDMIND_GNN_ENABLED=true + torch-geometric installed) ──
        # get_gnn_reranker() returns None whenever the flag is off or the dependency isn't
        # installed, so this is a true no-op for every default/existing configuration.
        try:
            if apply_post_rerankers:
                from engine.gnn_reranker import get_gnn_reranker
                gnn_reranker = get_gnn_reranker()
            else:
                gnn_reranker = None
            if gnn_reranker is not None:
                q_embedding = self.vector_engine.embedding_engine.embed(query_text)
                hybrid_results = gnn_reranker.rerank(
                    q_embedding,
                    hybrid_results,
                    self.graph_engine.graph_index,
                    top_k=rerank_pool,
                    sqlite_store=self.vector_engine.sqlite_store,
                )
        except Exception as e:
            logger.debug("GNN reranking skipped (%s)", type(e).__name__)

        # Query-local lexical reranking recovers exact source turns that are
        # present in the generated pool but under-ranked by corpus-global RRF.
        # It runs before the expensive cross-encoder so the neural stage sees a
        # stronger bounded candidate set.
        try:
            from config import settings as _cfg
            if apply_post_rerankers and _cfg.local_lexical_rerank_enabled and self.bm25_index is not None:
                from engine.lexical_reranker import rerank_with_query_local_lexical_rrf
                hybrid_results = rerank_with_query_local_lexical_rrf(
                    query_text,
                    hybrid_results,
                    self.bm25_index.tokenize,
                    pool_size=_cfg.local_lexical_rerank_pool_size,
                    lexical_weight=_cfg.local_lexical_rerank_weight,
                    rrf_k=get_rrf_k(),
                    document_term_cache=self._lexical_term_cache,
                )
        except Exception as e:
            logger.warning("Query-local lexical reranking skipped (%s)", type(e).__name__)

        # Cross-encoder reranking is the final text relevance stage. Only the
        # strongest fusion candidates enter the expensive model, and the API
        # contract is enforced here rather than leaking the full graph-expanded
        # pool to callers.
        hybrid_results.sort(key=lambda x: (-x["combined_score"], -x.get("vector_score", 0.0)))
        rerank_enabled = (
            apply_post_rerankers
            and self.reranker is not None
            and getattr(self.reranker, "enabled", True)
            and rerank_pool > 0
        )
        effective_rerank_pool = rerank_pool if rerank_enabled else top_k
        rerank_candidates = hybrid_results[:effective_rerank_pool]
        for candidate in rerank_candidates:
            candidate["rerank_attempted"] = rerank_enabled
            candidate["rerank_applied"] = False
        if rerank_enabled and rerank_candidates:
            rerank_candidates = self.reranker.rerank(
                query_text,
                rerank_candidates,
                top_k=None,
            )
            rerank_applied = any("rerank_score" in candidate for candidate in rerank_candidates)
            for candidate in rerank_candidates:
                candidate["rerank_applied"] = rerank_applied
                failure_type = candidate.get("rerank_failure_type")
                candidate["reasoning"] = (
                    f"{candidate.get('reasoning', '')}; "
                    f"rerank(attempted=true, applied={str(rerank_applied).lower()}, "
                    f"failure={failure_type or 'none'})"
                ).lstrip("; ")
        hybrid_results = rerank_candidates[:top_k]

        should_track_access = _cfg.access_tracking_enabled if track_access is None else track_access
        if should_track_access and hybrid_results:
            self.vector_engine.sqlite_store.record_access(
                [result["node_id"] for result in hybrid_results]
            )

        # Visual search candidates merge (Phase 7)
        if include_images and apply_post_rerankers:
            try:
                from engine.image_embedding import get_image_embedding_engine
                from api.dependencies import get_visual_store
                from storage.colbert_store import compute_maxsim
                import json

                img_engine = get_image_embedding_engine()
                visual_store = get_visual_store()
                if img_engine is not None and visual_store is not None:
                    query_patches = img_engine.embed_query(query_text)
                    if query_patches is not None and len(query_patches) > 0:
                        query_patches = np.asarray(query_patches, dtype=np.float32)
                        with self.vector_engine.sqlite_store._cursor() as cursor:
                            cursor.execute(
                                "SELECT id, text, metadata FROM nodes WHERE json_extract(metadata, '$.modality') = 'image' AND deleted_at IS NULL"
                            )
                            image_rows = cursor.fetchall()

                        visual_candidates = []
                        for row in image_rows:
                            nid = row["id"]
                            text = row["text"]
                            meta_str = row["metadata"]
                            try:
                                meta = json.loads(meta_str)
                            except Exception:
                                meta = {}

                            candidate_patches = visual_store.get(nid)
                            if candidate_patches is not None and len(candidate_patches) > 0:
                                sim = compute_maxsim(query_patches, candidate_patches)
                                visual_candidates.append({
                                    "node_id": nid,
                                    "text": text,
                                    "metadata": meta,
                                    "vector_score": sim,
                                    "graph_score": 0.0,
                                    "combined_score": sim,
                                    "reasoning": "visual_maxsim",
                                    "source": "visual_maxsim"
                                })

                        visual_candidates.sort(key=lambda x: -x["combined_score"])
                        results_with_visual = list(hybrid_results)
                        results_with_visual.extend(visual_candidates[:top_k])
                        hybrid_results = sorted(results_with_visual, key=lambda x: -x["combined_score"])[:top_k]
            except Exception as e:
                logger.warning("Visual memory search failed (%s)", type(e).__name__)

        query_time_ms = (time.perf_counter() - start_time) * 1000
        return hybrid_results, round(query_time_ms, 2), len(candidate_ids)

    def compare_search_modes(
        self,
        query_text: str,
        top_k: int = 10,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        anchor_nodes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare results across vector-only, graph-only, and hybrid modes.
        Useful for demonstrating hybrid advantages.

        Args:
            query_text: Search query
            top_k: Number of results per mode
            vector_weight: Weight for vector in hybrid
            graph_weight: Weight for graph in hybrid
            anchor_nodes: Anchor nodes for graph search

        Returns:
            Comparison results with all three modes
        """
        # Vector-only search
        vector_results, vector_time, vector_candidates = self.vector_engine.search(
            query_text=query_text,
            top_k=top_k
        )

        # Graph-only search (requires anchor)
        graph_results = []
        graph_time = 0.0
        graph_candidates = 0

        if anchor_nodes and not self.disable_graph_expansion:
            for anchor in anchor_nodes:
                results, time_ms, candidates = self.graph_engine.traverse(
                    start_id=anchor,
                    depth=2
                )
                graph_results.extend(results)
                graph_time += time_ms
                graph_candidates += candidates

            # Deduplicate
            seen: Set[str] = set()
            unique_graph = []
            for r in graph_results:
                if r["node_id"] not in seen:
                    seen.add(r["node_id"])
                    unique_graph.append(r)
            graph_results = unique_graph[:top_k]

        # Hybrid search
        hybrid_results, hybrid_time, hybrid_candidates = self.search(
            query_text=query_text,
            top_k=top_k,
            vector_weight=vector_weight,
            graph_weight=graph_weight,
            anchor_nodes=anchor_nodes
        )

        # Analyze overlap and unique finds
        vector_ids = {r["node_id"] for r in vector_results}
        graph_ids = {r["node_id"] for r in graph_results}
        hybrid_ids = {r["node_id"] for r in hybrid_results}

        return {
            "vector_only": {
                "results": vector_results,
                "query_time_ms": vector_time,
                "total_candidates": vector_candidates
            },
            "graph_only": {
                "results": graph_results,
                "query_time_ms": graph_time,
                "total_candidates": graph_candidates
            },
            "hybrid": {
                "results": hybrid_results,
                "query_time_ms": hybrid_time,
                "total_candidates": hybrid_candidates
            },
            "analysis": {
                "vector_unique": len(vector_ids - hybrid_ids),
                "graph_unique": len(graph_ids - hybrid_ids),
                "hybrid_unique": len(hybrid_ids - vector_ids - graph_ids),
                "overlap_all": len(vector_ids & graph_ids & hybrid_ids),
                "hybrid_combines_best": len(hybrid_ids & (vector_ids | graph_ids))
            }
        }
