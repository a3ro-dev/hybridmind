from fastapi import APIRouter, Depends
from typing import List, Dict, Any, Set
from pydantic import BaseModel
from api.dependencies import get_sqlite_store, get_vector_engine, get_hybrid_ranker
from storage.sqlite_store import SQLiteStore
from engine.vector_search import VectorSearchEngine
from engine.hybrid_ranker import HybridRanker
from engine.eval import EvalMetrics
import re

router = APIRouter(prefix="/comparison", tags=["Comparison"])

class EvalQuery(BaseModel):
    query: str
    ground_truth_ids: List[str]

class EvalSet(BaseModel):
    queries: List[EvalQuery]

def tokenize(text):
    return re.findall(r'\w+', text.lower())

@router.post("/effectiveness")
async def evaluate_effectiveness(
    eval_set: EvalSet,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_engine: VectorSearchEngine = Depends(get_vector_engine),
    hybrid_ranker: HybridRanker = Depends(get_hybrid_ranker)
) -> Dict[str, Any]:
    nodes = sqlite_store.list_nodes(limit=100000)
    node_ids = [n["id"] for n in nodes]
    documents = [tokenize(n["text"]) for n in nodes]
    
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(documents) if documents else None
    except ImportError:
        bm25 = None
        
    metrics = {
        "BM25": {"precision": [], "recall": [], "mrr": [], "ndcg": []},
        "Vector": {"precision": [], "recall": [], "mrr": [], "ndcg": []},
        "Hybrid": {"precision": [], "recall": [], "mrr": [], "ndcg": []}
    }
    
    for q in eval_set.queries:
        gt = set(q.ground_truth_ids)
        if not gt:
            continue
            
        # 1. BM25
        if bm25:
            tokenized_query = tokenize(q.query)
            scores = bm25.get_scores(tokenized_query)
            top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
            bm25_retrieved = [node_ids[i] for i in top_k_indices if scores[i] > 0]
            
            metrics["BM25"]["precision"].append(EvalMetrics.precision_at_k(bm25_retrieved, gt, 5))
            metrics["BM25"]["recall"].append(EvalMetrics.recall_at_k(bm25_retrieved, gt, 5))
            metrics["BM25"]["mrr"].append(EvalMetrics.mrr(bm25_retrieved, gt))
            metrics["BM25"]["ndcg"].append(EvalMetrics.ndcg_at_k(bm25_retrieved, gt, 5))
            
        # 2. Vector
        vector_results, _, _ = vector_engine.search(q.query, top_k=5)
        vector_retrieved = [r["node_id"] for r in vector_results]
        
        metrics["Vector"]["precision"].append(EvalMetrics.precision_at_k(vector_retrieved, gt, 5))
        metrics["Vector"]["recall"].append(EvalMetrics.recall_at_k(vector_retrieved, gt, 5))
        metrics["Vector"]["mrr"].append(EvalMetrics.mrr(vector_retrieved, gt))
        metrics["Vector"]["ndcg"].append(EvalMetrics.ndcg_at_k(vector_retrieved, gt, 5))
        
        # 3. Hybrid
        hybrid_results, _, _ = hybrid_ranker.search(
            query_text=q.query, 
            top_k=5, 
            vector_weight=0.7, 
            graph_weight=0.3
        )
        hybrid_retrieved = [r["node_id"] for r in hybrid_results]
        
        metrics["Hybrid"]["precision"].append(EvalMetrics.precision_at_k(hybrid_retrieved, gt, 5))
        metrics["Hybrid"]["recall"].append(EvalMetrics.recall_at_k(hybrid_retrieved, gt, 5))
        metrics["Hybrid"]["mrr"].append(EvalMetrics.mrr(hybrid_retrieved, gt))
        metrics["Hybrid"]["ndcg"].append(EvalMetrics.ndcg_at_k(hybrid_retrieved, gt, 5))
        
    # Aggregate
    result = {}
    for system, system_metrics in metrics.items():
        if not system_metrics["precision"]:
            result[system] = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
            continue
            
        result[system] = {
            "precision": sum(system_metrics["precision"]) / len(system_metrics["precision"]),
            "recall": sum(system_metrics["recall"]) / len(system_metrics["recall"]),
            "mrr": sum(system_metrics["mrr"]) / len(system_metrics["mrr"]),
            "ndcg": sum(system_metrics["ndcg"]) / len(system_metrics["ndcg"])
        }
        
    result["eval_note"] = "HybridMind is a subjective agent memory, not a web search engine. Ground truth is simulated via BM25 overlap for demo purposes."
    return result
