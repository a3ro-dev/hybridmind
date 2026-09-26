"""Tri-signal retrieval endpoint: independent dense, sparse and graph channels.

``POST /retrieve`` resolves a metadata scope to a derived scope-local corpus,
runs the requested channels, fuses them and returns ranked hits with per-channel
ranks, an optional packed evidence set, and an execution trace whose
``resolved_config_sha256`` covers every ranking knob.
"""

from datetime import timezone

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import DatabaseManager, get_db_manager
from config import settings
from engine.trisignal import ChannelUnavailable, RetrievalConfig
from models.retrieve import (
    EvidenceItemModel,
    EvidencePackModel,
    RetrievedHit,
    RetrieveRequest,
    RetrieveResponse,
)

router = APIRouter(tags=["Retrieve"])


def build_config(request: RetrieveRequest) -> RetrievalConfig:
    """Merge request fields over the server's ``trisignal_*`` defaults."""
    channels = tuple(request.channels or [c.strip() for c in settings.trisignal_channels.split(",") if c.strip()])
    evidence = request.evidence
    return RetrievalConfig(
        channels=channels,
        top_k=request.top_k,
        channel_k=request.channel_k or settings.trisignal_channel_k,
        fusion=request.fusion or settings.trisignal_fusion,
        rrf_k=settings.fusion_rrf_k if request.rrf_k is None else request.rrf_k,
        weights=tuple(sorted((request.weights or {}).items())),
        dense_mode=request.dense_mode or settings.trisignal_dense_mode,
        hnsw_ef_construction=settings.hnsw_ef_construction,
        hnsw_ef_search=settings.hnsw_ef_search,
        ann_audit=request.ann_audit,
        graph_method=request.graph_method or settings.trisignal_graph_method,
        graph_extractor=request.graph_extractor or settings.trisignal_graph_extractor,
        graph_query_extractor=settings.trisignal_graph_query_extractor,
        ppr_passage_seed_channel=request.ppr_passage_seed_channel,
        rerank_pool=request.rerank_pool,
        evidence_strategy=evidence.strategy if evidence else "turn",
        evidence_window=evidence.window if evidence else 1,
        evidence_lambda=evidence.propagation_lambda if evidence else 0.7,
        evidence_budget_tokens=evidence.budget_tokens if evidence else None,
        evidence_order=evidence.order if evidence else "chronological",
    )


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    request: RetrieveRequest,
    manager: DatabaseManager = Depends(get_db_manager),
) -> RetrieveResponse:
    config = build_config(request)
    try:
        config.validate()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must contain non-whitespace text")
    if config.rerank_pool and len(request.query) > settings.reranker_max_query_chars:
        raise HTTPException(
            status_code=422,
            detail=f"query exceeds reranker_max_query_chars ({settings.reranker_max_query_chars}) with rerank_pool > 0",
        )

    store = manager.sqlite_store
    generation = store.get_corpus_generation()
    as_of = request.as_of.astimezone(timezone.utc).isoformat() if request.as_of else None
    try:
        scope = manager.scope_registry.get(request.scope, as_of=as_of)
        result = manager.trisignal.retrieve(scope, request.query, config, query_keys=request.query_keys)
    except ChannelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    # Any other exception (corrupt persisted vectors, provider shape errors)
    # is a server fault and surfaces as 500, never as a client error.
    if store.get_corpus_generation() != generation:
        raise HTTPException(status_code=409, detail="corpus changed during retrieval; retry against a stable generation")
    result.trace["corpus_generation"] = generation

    evidence = None
    if result.evidence is not None:
        corpus = scope.corpus
        evidence = EvidencePackModel(
            strategy=result.evidence.strategy,
            budget_tokens=result.evidence.budget_tokens,
            packed_tokens=result.evidence.packed_tokens,
            dropped_hits=result.evidence.dropped_hits,
            items=[
                EvidenceItemModel(
                    node_id=item.node_id,
                    evidence_id=item.evidence_id,
                    role=item.role,
                    rank=item.rank,
                    score=item.score,
                    expanded_from=item.expanded_from,
                    tokens=item.tokens,
                    text=corpus.turn(item.node_id).text,
                )
                for item in result.evidence.items
            ],
        )
    return RetrieveResponse(
        hits=[
            RetrievedHit(
                node_id=h.node_id,
                evidence_id=h.evidence_id,
                text=h.text,
                metadata=h.metadata,
                rank=h.rank,
                score=h.score,
                sources=h.sources,
                channel_ranks=h.channel_ranks,
                channel_scores=h.channel_scores,
                rerank_score=h.rerank_score,
            )
            for h in result.hits
        ],
        evidence=evidence,
        trace=result.trace,
    )
