"""
Bulk operations API endpoints for HybridMind.
Fast batch import for nodes and edges.
Includes LLM-powered unstructured data processing.
"""

import logging
import math
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from api.dependencies import (
    get_sqlite_store,
    get_vector_index,
    get_graph_index,
    get_embedding_engine,
    get_db_manager,
    get_bm25_index,
    coordinate_mutation,
)
from storage.sqlite_store import SQLiteStore
from storage.vector_index import VectorIndex
from storage.bm25_index import sparse_document_text
from typing import Any as _BM25AnyType
from storage.graph_index import GraphIndex
from engine.embedding import EmbeddingEngine, validate_embedding_4096
from engine.cache import invalidate_cache
from engine.edge_inference import run_auto_edge_inference
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bulk", tags=["Bulk Operations"])
MAX_BULK_TOTAL_CHARS = 1_000_000


def _node_payload_chars(nodes: List["BulkNodeCreate"]) -> int:
    return sum(
        len(node.text)
        + len(json.dumps(node.metadata, ensure_ascii=False, separators=(",", ":"), default=str))
        for node in nodes
    )


def _edge_payload_chars(edges: List["BulkEdgeCreate"]) -> int:
    return sum(
        len(edge.source_id) + len(edge.target_id) + len(edge.type)
        + len(json.dumps(edge.metadata, ensure_ascii=False, separators=(",", ":"), default=str))
        for edge in edges
    )

# ==================== Request/Response Models ====================

class BulkNodeCreate(BaseModel):
    """Single node in bulk create request."""
    id: Optional[str] = Field(default=None, description="Optional custom ID")
    text: str = Field(..., min_length=1, max_length=50000)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BulkNodesRequest(BaseModel):
    """Bulk node creation request."""
    nodes: List[BulkNodeCreate] = Field(..., min_length=1, max_length=1000)
    generate_embeddings: bool = Field(
        default=True,
        description="Generate embeddings for nodes"
    )

    @model_validator(mode="after")
    def embeddings_are_mandatory(self):
        if not self.generate_embeddings:
            raise ValueError(
                "HybridMind requires exact 4096-dimensional embeddings for every node"
            )
        if _node_payload_chars(self.nodes) > MAX_BULK_TOTAL_CHARS:
            raise ValueError("Bulk node payload exceeds the total character limit")
        ids = [node.id for node in self.nodes if node.id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate explicit node IDs are not allowed")
        return self


class BulkEdgeCreate(BaseModel):
    """Single edge in bulk create request."""
    id: Optional[str] = Field(default=None, description="Optional custom ID")
    source_id: str
    target_id: str
    type: str = Field(..., min_length=1)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BulkEdgesRequest(BaseModel):
    """Bulk edge creation request."""
    edges: List[BulkEdgeCreate] = Field(..., min_length=1, max_length=5000)
    skip_validation: bool = Field(
        default=False,
        description="Deprecated compatibility flag; endpoint existence is always validated"
    )

    @model_validator(mode="after")
    def edge_ids_are_unique_and_finite(self):
        ids = [edge.id for edge in self.edges if edge.id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate explicit edge IDs are not allowed")
        if any(not math.isfinite(edge.weight) for edge in self.edges):
            raise ValueError("Edge weights must be finite")
        if _edge_payload_chars(self.edges) > MAX_BULK_TOTAL_CHARS:
            raise ValueError("Bulk edge payload exceeds the total character limit")
        return self


class BulkResult(BaseModel):
    """Result of bulk operation."""
    success: bool
    created: int
    failed: int
    errors: List[str]
    elapsed_ms: float


class BulkImportRequest(BaseModel):
    """Combined bulk import of nodes and edges."""
    nodes: List[BulkNodeCreate] = Field(default_factory=list)
    edges: List[BulkEdgeCreate] = Field(default_factory=list)
    generate_embeddings: bool = Field(default=True)

    @model_validator(mode="after")
    def embeddings_are_mandatory(self):
        if not self.generate_embeddings:
            raise ValueError(
                "HybridMind requires exact 4096-dimensional embeddings for every node"
            )
        if (
            _node_payload_chars(self.nodes) + _edge_payload_chars(self.edges)
            > MAX_BULK_TOTAL_CHARS
        ):
            raise ValueError("Bulk import payload exceeds the total character limit")
        node_ids = [node.id for node in self.nodes if node.id is not None]
        edge_ids = [edge.id for edge in self.edges if edge.id is not None]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Duplicate explicit node IDs are not allowed")
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("Duplicate explicit edge IDs are not allowed")
        return self


class BulkImportResult(BaseModel):
    """Result of combined bulk import."""
    nodes: BulkResult
    edges: BulkResult
    total_elapsed_ms: float


class UnstructuredDataRequest(BaseModel):
    """Request for processing unstructured data via LLM."""
    text: str = Field(..., min_length=10, max_length=12000, description="Raw unstructured text to process")
    api_key: Optional[str] = Field(default=None, description="Deprecated; provider credentials come from config.py")
    model: Optional[str] = Field(default=None, description="Deprecated; provider models come from config.py")


class UnstructuredDataResult(BaseModel):
    """Result of unstructured data processing."""
    success: bool
    summary: str
    nodes_created: int
    edges_created: int
    nodes_failed: int
    edges_failed: int
    extracted_entities: List[Dict[str, Any]]
    errors: List[str]
    elapsed_ms: float


def _delete_sql_nodes(sqlite_store: SQLiteStore, node_ids: List[str]) -> None:
    if not node_ids:
        return
    placeholders = ",".join("?" for _ in node_ids)
    with sqlite_store._cursor() as cursor:
        # Compensation is rollback, not a historical user event. Failed
        # attempts must leave neither current rows nor bitemporal versions.
        cursor.execute(
            f"DELETE FROM node_versions WHERE node_id IN ({placeholders})",
            tuple(node_ids),
        )
        cursor.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", tuple(node_ids))


def _rollback_nodes(
    sqlite_store: SQLiteStore,
    vector_index: VectorIndex,
    graph_index: GraphIndex,
    bm25_index: _BM25AnyType,
    node_ids: List[str],
) -> None:
    """Best-effort derived-index cleanup followed by authoritative SQL removal."""
    cleanup_failures = []
    for node_id in reversed(node_ids):
        for label, operation in (
            ("bm25", lambda nid=node_id: bm25_index.remove(nid)),
            ("vector", lambda nid=node_id: vector_index.remove(nid)),
            ("graph", lambda nid=node_id: graph_index.remove_node(nid)),
        ):
            try:
                operation()
            except Exception as exc:
                cleanup_failures.append((label, type(exc).__name__))
    _delete_sql_nodes(sqlite_store, node_ids)
    if cleanup_failures:
        logger.error("Bulk node compensation had derived-index cleanup failures: %s", cleanup_failures)
    try:
        _rebuild_indexes_from_sql(sqlite_store, vector_index, graph_index, bm25_index)
    except Exception as exc:
        logger.critical("Bulk node compensation rebuild failed type=%s", type(exc).__name__)


def _rollback_edges(
    sqlite_store: SQLiteStore,
    graph_index: GraphIndex,
    edge_ids: List[str],
) -> None:
    for edge_id in reversed(edge_ids):
        try:
            graph_index.remove_edge_by_id(edge_id)
        except Exception as exc:
            logger.error("Bulk edge graph compensation failed type=%s", type(exc).__name__)
        try:
            sqlite_store.delete_edge(edge_id)
        except Exception as exc:
            logger.error("Bulk edge SQL compensation failed type=%s", type(exc).__name__)
    try:
        graph_index.rebuild_from_edges(sqlite_store.get_all_edges())
        for node in sqlite_store.list_nodes(limit=1_000_000):
            if not graph_index.has_node(node["id"]):
                graph_index.add_node(node["id"])
        for node in sqlite_store.list_nodes(limit=1_000_000, include_archived=True):
            if node.get("archived_at"):
                graph_index.remove_node(node["id"])
    except Exception as exc:
        logger.critical("Bulk edge compensation rebuild failed type=%s", type(exc).__name__)


def _create_edge_consistently(
    sqlite_store: SQLiteStore,
    graph_index: GraphIndex,
    *,
    edge_id: str,
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    sqlite_store.create_edge(
        edge_id=edge_id,
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        weight=weight,
        metadata=metadata,
    )
    try:
        graph_index.add_edge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {},
        )
    except Exception:
        try:
            graph_index.remove_edge_by_id(edge_id)
        except Exception:
            pass
        sqlite_store.delete_edge(edge_id)
        raise


def _reconcile_created_node_edges(
    sqlite_store: SQLiteStore,
    graph_index: GraphIndex,
    node_ids: List[str],
) -> None:
    """Ensure auto/session edges committed in SQL are reachable in the graph."""
    seen = set()
    for node_id in node_ids:
        for edge in sqlite_store.get_node_edges(node_id):
            edge_id = edge["id"]
            if edge_id in seen or graph_index.get_edge_by_id(edge_id) is not None:
                continue
            seen.add(edge_id)
            graph_index.add_edge(
                edge_id=edge_id,
                source_id=edge["source_id"],
                target_id=edge["target_id"],
                edge_type=edge["type"],
                weight=float(edge["weight"]),
            )


def _rebuild_indexes_from_sql(
    sqlite_store: SQLiteStore,
    vector_index: VectorIndex,
    graph_index: GraphIndex,
    bm25_index: _BM25AnyType,
) -> None:
    """Rebuild primary derived indexes from the authoritative SQLite state."""
    embeddings = sqlite_store.get_all_node_embeddings(include_archived=False)
    vector_index.rebuild_from_embeddings(embeddings)
    edges = sqlite_store.get_all_edges()
    graph_index.rebuild_from_edges(edges)
    nodes = sqlite_store.list_nodes(limit=1_000_000)
    for node in nodes:
        if not graph_index.has_node(node["id"]):
            graph_index.add_node(node["id"])
    archived_nodes = sqlite_store.list_nodes(limit=1_000_000, include_archived=True)
    for node in archived_nodes:
        if node.get("archived_at"):
            graph_index.remove_node(node["id"])
    if hasattr(bm25_index, "rebuild_from_nodes"):
        bm25_index.rebuild_from_nodes(nodes)
    else:
        bm25_index.clear()
        bm25_index.add_batch([
            (
                node["id"],
                sparse_document_text(node["text"], node.get("metadata")),
            )
            for node in nodes
        ])


def _rebuild_primary_indexes(db_manager) -> None:
    _rebuild_indexes_from_sql(
        db_manager.sqlite_store,
        db_manager.vector_index,
        db_manager.graph_index,
        db_manager.bm25_index,
    )


# ==================== Endpoints ====================

@router.post("/nodes", response_model=BulkResult)
async def bulk_create_nodes(
    request: BulkNodesRequest,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    graph_index: GraphIndex = Depends(get_graph_index),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    bm25_index: _BM25AnyType = Depends(get_bm25_index),
    mutation_guard: None = Depends(coordinate_mutation),
):
    """
    Bulk create nodes with optional embedding generation.
    
    Features:
    - Batch embedding generation for efficiency
    - Batch vector index insertion
    - Automatic ID generation if not provided
    
    Maximum 1000 nodes per request.
    """
    start_time = time.perf_counter()
    nodes_to_create = []
    for node in request.nodes:
        node_id = node.id or f"node_{uuid.uuid4().hex[:12]}"
        if sqlite_store.get_node(node_id):
            raise HTTPException(status_code=409, detail="One or more node IDs already exist")
        metadata = dict(node.metadata or {})
        try:
            event_time, valid_from, valid_until = SQLiteStore.normalize_temporal_fields(
                event_time=(
                    metadata.get("event_time")
                    or metadata.get("date")
                    or metadata.get("timestamp")
                ),
                valid_from=metadata.get("valid_from"),
                valid_until=metadata.get("valid_until"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if event_time is not None:
            metadata["event_time"] = event_time
            if metadata.get("date"):
                metadata["date"] = event_time
            if metadata.get("timestamp"):
                metadata["timestamp"] = event_time
        if valid_from is not None:
            metadata["valid_from"] = valid_from
        if valid_until is not None:
            metadata["valid_until"] = valid_until
        try:
            confidence = float(metadata.get("confidence", 1.0))
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Node confidence must be numeric")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise HTTPException(status_code=422, detail="Node confidence must be finite and between 0 and 1")
        nodes_to_create.append({
            "id": node_id,
            "text": node.text,
            "metadata": metadata,
            "confidence": confidence,
            "event_time": event_time,
            "valid_from": valid_from,
            "valid_until": valid_until,
        })

    import asyncio
    import numpy as np
    import re as _re
    prefix_re = _re.compile(r'^(\[DATE:[^\]]*\]\s*)?(\[SPEAKER:[^\]]*\]\s*)?')
    texts = [prefix_re.sub('', node["text"], count=1).strip() for node in nodes_to_create]
    try:
        raw_embeddings = await asyncio.to_thread(
            embedding_engine.embed_batch,
            texts,
            True,
            max(1, settings.embedding_batch_size),
            False,
        )
        embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        if embeddings.shape != (len(texts), 4096) or not np.all(np.isfinite(embeddings)):
            raise ValueError("invalid bulk embedding batch")
        validated_embeddings = [
            validate_embedding_4096(embeddings[index], label="bulk embedding")
            for index in range(len(nodes_to_create))
        ]
    except Exception as exc:
        logger.error("Bulk embedding preflight failed type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Exact finite 4096-dimensional bulk embedding failed; no fallback was used",
        ) from exc

    created_node_ids: List[str] = []
    colbert_store = None
    colbert_node_ids: List[str] = []
    try:
        for node_data, embedding in zip(nodes_to_create, validated_embeddings):
            metadata = node_data["metadata"]
            sqlite_store.create_node(
                node_id=node_data["id"],
                text=node_data["text"],
                metadata=metadata,
                embedding=embedding,
                raw_embedding=embedding,
                event_time=node_data["event_time"],
                valid_from=node_data["valid_from"],
                valid_until=node_data["valid_until"],
                memory_kind=metadata.get("memory_kind"),
                confidence=node_data["confidence"],
            )
            created_node_ids.append(node_data["id"])
            graph_index.add_node(node_data["id"])
        vector_batch = [
            (node_data["id"], embedding)
            for node_data, embedding in zip(nodes_to_create, validated_embeddings)
        ]
        vector_index.add_batch(vector_batch)
        bm25_index.add_batch([
            (
                node["id"],
                sparse_document_text(node["text"], node.get("metadata")),
            )
            for node in nodes_to_create
        ])

        # ColBERT is part of the active retrieval contract when enabled.  It is
        # therefore a required stage, not a best-effort post-success side effect.
        from storage.colbert_store import colbert_enabled, maybe_store_colbert
        if colbert_enabled():
            from api.dependencies import get_colbert_store
            colbert_store = get_colbert_store()
            if colbert_store is None:
                raise RuntimeError("enabled ColBERT store is unavailable")
            for node_data in nodes_to_create:
                colbert_node_ids.append(node_data["id"])
                if not maybe_store_colbert(
                    node_data["id"], node_data["text"], embedding_engine, colbert_store
                ):
                    raise RuntimeError("enabled ColBERT indexing failed")

        sessions: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for node_data in nodes_to_create:
            metadata = node_data["metadata"]
            session_id = metadata.get("sessionId") or metadata.get("session_id")
            if session_id:
                container = metadata.get("containerTag") or metadata.get("container_tag") or "__default__"
                turn_index = metadata.get("turn_index")
                try:
                    normalized_turn_index = (
                        int(turn_index) if turn_index is not None else None
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError("turn_index must be an integer") from exc
                sessions.setdefault((str(container), str(session_id)), []).append(
                    {
                        "id": node_data["id"],
                        "turn_index": normalized_turn_index,
                    }
                )
        for session_nodes in sessions.values():
            if all(node["turn_index"] is not None for node in session_nodes):
                indices = [node["turn_index"] for node in session_nodes]
                if len(indices) != len(set(indices)):
                    raise ValueError("A session contains duplicate turn_index values")
                session_nodes.sort(key=lambda node: (node["turn_index"], node["id"]))
            session_node_ids = [node["id"] for node in session_nodes]
            for source_id, target_id in zip(session_node_ids, session_node_ids[1:]):
                for edge_type, weight in (("next_turn", 1.0), ("same_session", 0.5)):
                    _create_edge_consistently(
                        sqlite_store,
                        graph_index,
                        edge_id=f"edge_{uuid.uuid4().hex[:12]}",
                        source_id=source_id,
                        target_id=target_id,
                        edge_type=edge_type,
                        weight=weight,
                    )

        for node_data, embedding in zip(nodes_to_create, validated_embeddings):
            run_auto_edge_inference(
                node_id=node_data["id"],
                embedding=embedding,
                node_metadata=node_data["metadata"],
                node_text=node_data["text"],
                vector_index=vector_index,
                sqlite_store=sqlite_store,
                graph_index=graph_index,
                event_time=node_data["metadata"].get("event_time")
                or node_data["metadata"].get("date"),
            )
        _reconcile_created_node_edges(sqlite_store, graph_index, created_node_ids)
    except Exception as exc:
        logger.error("Bulk node mutation failed; compensating type=%s", type(exc).__name__)
        if colbert_store is not None:
            for node_id in colbert_node_ids:
                try:
                    colbert_store.remove(node_id)
                except Exception as cleanup_exc:
                    logger.error("ColBERT compensation failed type=%s", type(cleanup_exc).__name__)
        _rollback_nodes(
            sqlite_store, vector_index, graph_index, bm25_index, created_node_ids
        )
        invalidate_cache()
        raise HTTPException(
            status_code=500,
            detail="Bulk node creation failed; created rows were rolled back",
        ) from exc

    # Invalidate cache
    invalidate_cache()
    
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info("Bulk created %d nodes in %.0fms", len(created_node_ids), elapsed)
    
    return BulkResult(
        success=True,
        created=len(created_node_ids),
        failed=0,
        errors=[],
        elapsed_ms=round(elapsed, 2)
    )


@router.post("/edges", response_model=BulkResult)
async def bulk_create_edges(
    request: BulkEdgesRequest,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    graph_index: GraphIndex = Depends(get_graph_index),
    mutation_guard: None = Depends(coordinate_mutation),
):
    """
    Bulk create edges between existing nodes.
    
    Features:
    - Optional node existence validation (disable for faster import)
    - Automatic ID generation if not provided
    
    Maximum 5000 edges per request.
    """
    start_time = time.perf_counter()
    prepared = []
    for edge in request.edges:
        edge_id = edge.id or f"edge_{uuid.uuid4().hex[:12]}"
        if sqlite_store.get_edge(edge_id):
            raise HTTPException(status_code=409, detail="One or more edge IDs already exist")
        prepared.append((edge_id, edge))

    # SQL foreign-key enforcement is not a substitute for a useful API error;
    # validate even when legacy callers request skip_validation.
    endpoint_ids = {endpoint for _, edge in prepared for endpoint in (edge.source_id, edge.target_id)}
    missing = [node_id for node_id in endpoint_ids if not sqlite_store.get_node(node_id)]
    if missing:
        raise HTTPException(status_code=422, detail="One or more edge endpoints do not exist")

    created_edge_ids: List[str] = []
    try:
        for edge_id, edge in prepared:
            _create_edge_consistently(
                sqlite_store,
                graph_index,
                edge_id=edge_id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                edge_type=edge.type,
                weight=edge.weight,
                metadata=edge.metadata,
            )
            created_edge_ids.append(edge_id)
    except Exception as exc:
        logger.error("Bulk edge mutation failed; compensating type=%s", type(exc).__name__)
        _rollback_edges(sqlite_store, graph_index, created_edge_ids)
        invalidate_cache()
        raise HTTPException(
            status_code=500,
            detail="Bulk edge creation failed; created rows were rolled back",
        ) from exc
    
    # Invalidate cache
    invalidate_cache()
    
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info("Bulk created %d edges in %.0fms", len(created_edge_ids), elapsed)
    
    return BulkResult(
        success=True,
        created=len(created_edge_ids),
        failed=0,
        errors=[],
        elapsed_ms=round(elapsed, 2)
    )


@router.post("/import", response_model=BulkImportResult)
async def bulk_import(
    request: BulkImportRequest,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    graph_index: GraphIndex = Depends(get_graph_index),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    bm25_index: _BM25AnyType = Depends(get_bm25_index),
    mutation_guard: None = Depends(coordinate_mutation),
):
    """
    Combined bulk import of nodes and edges.
    
    Imports nodes first, then edges. Useful for loading
    complete knowledge graphs in a single request.
    """
    total_start = time.perf_counter()
    prepared_nodes = [
        node if node.id is not None else node.model_copy(update={"id": f"node_{uuid.uuid4().hex[:12]}"})
        for node in request.nodes
    ]

    node_result = await bulk_create_nodes(
        BulkNodesRequest(
            nodes=prepared_nodes,
            generate_embeddings=request.generate_embeddings
        ),
        sqlite_store=sqlite_store,
        vector_index=vector_index,
        graph_index=graph_index,
        embedding_engine=embedding_engine,
        bm25_index=bm25_index,
    ) if request.nodes else BulkResult(
        success=True, created=0, failed=0, errors=[], elapsed_ms=0
    )
    
    try:
        edge_result = await bulk_create_edges(
            BulkEdgesRequest(edges=request.edges, skip_validation=False),
            sqlite_store=sqlite_store,
            graph_index=graph_index,
        ) if request.edges else BulkResult(
            success=True, created=0, failed=0, errors=[], elapsed_ms=0
        )
    except Exception as exc:
        _rollback_nodes(
            sqlite_store,
            vector_index,
            graph_index,
            bm25_index,
            [node.id for node in prepared_nodes if node.id is not None],
        )
        invalidate_cache()
        logger.error("Combined bulk import failed; nodes compensated type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Bulk import failed; all created data was rolled back",
        ) from exc
    
    total_elapsed = (time.perf_counter() - total_start) * 1000
    
    logger.info(
        f"Bulk import complete: {node_result.created} nodes, "
        f"{edge_result.created} edges in {total_elapsed:.0f}ms"
    )
    
    return BulkImportResult(
        nodes=node_result,
        edges=edge_result,
        total_elapsed_ms=round(total_elapsed, 2)
    )


@router.post("/unstructured", response_model=UnstructuredDataResult)
async def process_unstructured_data(
    request: UnstructuredDataRequest,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    graph_index: GraphIndex = Depends(get_graph_index),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    bm25_index: _BM25AnyType = Depends(get_bm25_index),
    mutation_guard: None = Depends(coordinate_mutation),
):
    """
    Process unstructured text using LLM and extract knowledge graph.
    
    Uses the centralized, policy-gated LLM provider to:
    - Extract entities, concepts, and facts from raw text
    - Create structured nodes with rich metadata
    - Identify and create relationships between nodes
    
    Perfect for ingesting:
    - Wikipedia articles
    - Research papers
    - Documentation
    - Any large text content
    """
    start_time = time.perf_counter()
    try:
        from engine.llm import LLMEngine
        llm = LLMEngine()
    except Exception as exc:
        logger.error("Unstructured LLM initialization failed type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="The configured unstructured-ingest provider is unavailable",
        ) from exc

    try:
        import asyncio
        extracted = await asyncio.to_thread(llm.process_unstructured, request.text)
    except Exception as exc:
        logger.error("Unstructured provider processing failed type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="Unstructured extraction failed validation or provider processing",
        ) from exc

    if not isinstance(extracted, dict):
        raise HTTPException(status_code=502, detail="Unstructured extraction returned an invalid document")
    summary = extracted.get("summary", "")
    raw_nodes = extracted.get("nodes")
    raw_edges = extracted.get("edges")
    if not isinstance(summary, str) or not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise HTTPException(status_code=502, detail="Unstructured extraction returned an invalid schema")
    if not raw_nodes or len(raw_nodes) > 1000 or len(raw_edges) > 5000:
        raise HTTPException(status_code=502, detail="Unstructured extraction exceeded safe resource limits")

    node_id_map: Dict[int, str] = {}
    prepared_nodes: List[BulkNodeCreate] = []
    for index, node_data in enumerate(raw_nodes):
        if not isinstance(node_data, dict):
            raise HTTPException(status_code=502, detail="Unstructured extraction returned an invalid node")
        node_text = node_data.get("text")
        metadata = node_data.get("metadata", {})
        if not isinstance(node_text, str) or not 10 <= len(node_text) <= 50000 or not isinstance(metadata, dict):
            raise HTTPException(status_code=502, detail="Unstructured extraction returned an invalid node")
        node_id = f"node_{uuid.uuid4().hex[:12]}"
        node_id_map[index] = node_id
        enriched_metadata = dict(metadata)
        enriched_metadata["source"] = "llm_extraction"
        enriched_metadata["summary_context"] = summary[:200]
        prepared_nodes.append(
            BulkNodeCreate(id=node_id, text=node_text, metadata=enriched_metadata)
        )

    prepared_edges: List[BulkEdgeCreate] = []
    for edge_data in raw_edges:
        if not isinstance(edge_data, dict):
            raise HTTPException(status_code=502, detail="Unstructured extraction returned an invalid edge")
        source_index = edge_data.get("source_index")
        target_index = edge_data.get("target_index")
        if (
            isinstance(source_index, bool)
            or isinstance(target_index, bool)
            or not isinstance(source_index, int)
            or not isinstance(target_index, int)
            or source_index not in node_id_map
            or target_index not in node_id_map
        ):
            raise HTTPException(status_code=502, detail="Unstructured extraction returned an invalid edge endpoint")
        edge_type = edge_data.get("type", "relates_to")
        weight = edge_data.get("weight", 0.5)
        if not isinstance(edge_type, str) or not edge_type.strip():
            raise HTTPException(status_code=502, detail="Unstructured extraction returned an invalid edge type")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
            raise HTTPException(status_code=502, detail="Unstructured extraction returned an invalid edge weight")
        if not 0.0 <= float(weight) <= 1.0:
            raise HTTPException(status_code=502, detail="Unstructured extraction returned an invalid edge weight")
        prepared_edges.append(BulkEdgeCreate(
            id=f"edge_{uuid.uuid4().hex[:12]}",
            source_id=node_id_map[source_index],
            target_id=node_id_map[target_index],
            type=edge_type,
            weight=float(weight),
            metadata={"reasoning": str(edge_data.get("reasoning", ""))[:2000]},
        ))

    node_result = await bulk_create_nodes(
        BulkNodesRequest(nodes=prepared_nodes, generate_embeddings=True),
        sqlite_store=sqlite_store,
        vector_index=vector_index,
        graph_index=graph_index,
        embedding_engine=embedding_engine,
        bm25_index=bm25_index,
    )
    try:
        edge_result = await bulk_create_edges(
            BulkEdgesRequest(edges=prepared_edges, skip_validation=False),
            sqlite_store=sqlite_store,
            graph_index=graph_index,
        ) if prepared_edges else BulkResult(
            success=True, created=0, failed=0, errors=[], elapsed_ms=0
        )
    except Exception as exc:
        _rollback_nodes(
            sqlite_store,
            vector_index,
            graph_index,
            bm25_index,
            list(node_id_map.values()),
        )
        invalidate_cache()
        logger.error("Unstructured edge commit failed; nodes compensated type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Unstructured ingest failed; all created data was rolled back",
        ) from exc

    elapsed = (time.perf_counter() - start_time) * 1000
    extracted_entities = [
        {
            "node_id": node_id_map[index],
            "text_preview": raw_nodes[index]["text"][:100],
            "metadata": raw_nodes[index].get("metadata", {}),
        }
        for index in range(min(20, len(raw_nodes)))
    ]
    logger.info(
        "Unstructured import committed nodes=%d edges=%d in %.0fms",
        node_result.created,
        edge_result.created,
        elapsed,
    )
    return UnstructuredDataResult(
        success=True,
        summary=summary,
        nodes_created=node_result.created,
        edges_created=edge_result.created,
        nodes_failed=0,
        edges_failed=0,
        extracted_entities=extracted_entities,
        errors=[],
        elapsed_ms=round(elapsed, 2),
    )


@router.delete("/clear", response_model=dict)
async def clear_all_data(mutation_guard: None = Depends(coordinate_mutation)):
    """
    Clear all data from the database.
    
    **WARNING**: This permanently deletes all nodes, edges, and indexes.
    Use with caution.
    """
    start_time = time.perf_counter()
    db_manager = None
    try:
        db_manager = get_db_manager()
        
        # Get counts before clearing
        stats = db_manager.get_stats()
        nodes_count = stats["total_nodes"]
        edges_count = stats["total_edges"]
        
        # Keep the SQL transaction open until all primary indexes clear. If an
        # index operation fails, SQL rolls back and the indexes are rebuilt.
        with db_manager.sqlite_store._cursor() as cursor:
            cursor.execute("DELETE FROM edges")
            # Clear is an erasure boundary, unlike ordinary lifecycle soft
            # deletion. Remove immutable history before current rows.
            cursor.execute("DELETE FROM node_versions")
            cursor.execute("DELETE FROM nodes")
            db_manager.vector_index.clear()
            db_manager.graph_index.clear()
            db_manager.bm25_index.clear()
        
        # Invalidate cache
        invalidate_cache()
        
        elapsed = (time.perf_counter() - start_time) * 1000
        
        return {
            "success": True,
            "deleted_nodes": nodes_count,
            "deleted_edges": edges_count,
            "elapsed_ms": round(elapsed, 2)
        }
        
    except Exception as exc:
        logger.error("Bulk clear failed type=%s", type(exc).__name__)
        if db_manager is not None:
            try:
                _rebuild_primary_indexes(db_manager)
            except Exception as rebuild_exc:
                logger.critical("Bulk clear recovery failed type=%s", type(rebuild_exc).__name__)
        raise HTTPException(status_code=500, detail="Failed to clear data safely") from exc

