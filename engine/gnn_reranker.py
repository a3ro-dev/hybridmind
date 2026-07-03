"""
GNN-based reranker for HybridMind (opt-in, off by default).

Enable with: HYBRIDMIND_GNN_ENABLED=true
Requires:    pip install torch-geometric

Architecture: GraphSAGE (2-layer) over the typed HybridMind graph.
The GNN produces node embeddings that capture multi-hop relational context;
candidates are re-scored by cosine similarity to the query GNN embedding.

Training:  scripts/train_gnn.py  (RunPod)
Inference: runs CPU-only if no GPU is available; gracefully falls back to
           standard BFS proximity scoring when disabled or torch-geometric
           is not installed.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GNNReranker:
    """
    GNN reranker that replaces BFS proximity with learned graph embeddings.

    Usage:
        reranker = GNNReranker(checkpoint_path="models/gnn.pt")
        reranked = reranker.rerank(query_text, candidates, graph_index, top_k=10)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self._model = None
        self._available = False
        self._try_init()

    def _try_init(self):
        try:
            import torch  # noqa: F401
            import torch_geometric  # noqa: F401
            self._available = True
            if self.checkpoint_path:
                self._load_checkpoint()
            else:
                logger.info(
                    "GNNReranker: no checkpoint path set. "
                    "Train one with scripts/train_gnn.py, then set the path."
                )
        except ImportError:
            logger.info(
                "GNNReranker: torch-geometric not installed — "
                "falling back to BFS proximity. "
                "Install with: pip install torch-geometric"
            )

    def _load_checkpoint(self):
        import torch
        from torch_geometric.nn import SAGEConv
        import torch.nn as nn

        class GraphSAGE(nn.Module):
            def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
                super().__init__()
                self.convs = nn.ModuleList()
                self.convs.append(SAGEConv(in_dim, hidden_dim))
                for _ in range(num_layers - 2):
                    self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                self.convs.append(SAGEConv(hidden_dim, out_dim))

            def forward(self, x, edge_index):
                import torch.nn.functional as F
                for conv in self.convs[:-1]:
                    x = F.relu(conv(x, edge_index))
                return self.convs[-1](x, edge_index)

        device = self.device or "cpu"
        try:
            ckpt = torch.load(self.checkpoint_path, map_location=device)
            in_dim = ckpt.get("in_dim", 1024)
            hidden_dim = ckpt.get("hidden_dim", self.hidden_dim)
            out_dim = ckpt.get("out_dim", hidden_dim)
            model = GraphSAGE(in_dim, hidden_dim, out_dim, self.num_layers).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self._model = model
            self._device = device
            logger.info(f"GNNReranker: loaded checkpoint from {self.checkpoint_path}")
        except Exception as e:
            logger.warning(f"GNNReranker: checkpoint load failed: {e}")

    def is_available(self) -> bool:
        return self._available and self._model is not None

    def rerank(
        self,
        query_embedding,
        candidates: List[Dict[str, Any]],
        graph_index,
        top_k: Optional[int] = None,
        sqlite_store: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank candidates using GNN node embeddings.
        Falls back to original order if model is not available.

        Args:
            query_embedding: dense query vector (e.g. from embedding_engine.embed()).
            candidates: hybrid_ranker result dicts (must have "node_id", "combined_score").
            graph_index: GraphIndex instance (for adjacency).
            top_k: slice size for the returned list.
            sqlite_store: SQLiteStore instance used to fetch real per-node embeddings
                for the GNN input features. Falls back to zero vectors (structure-only,
                no node-identity signal) for nodes with no stored embedding or when
                sqlite_store is not provided.
        """
        if not self.is_available():
            return candidates[:top_k] if top_k else candidates

        try:
            return self._gnn_rerank(query_embedding, candidates, graph_index, top_k, sqlite_store)
        except Exception as e:
            logger.warning(f"GNNReranker: inference failed, falling back: {e}")
            return candidates[:top_k] if top_k else candidates

    def _gnn_rerank(self, query_embedding, candidates, graph_index, top_k, sqlite_store=None):
        import torch
        import numpy as np

        node_ids = [c["node_id"] for c in candidates]
        # Build subgraph edge index
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        edge_src, edge_dst = [], []
        for src_id in node_ids:
            if not graph_index.has_node(src_id):
                continue
            for nb in graph_index.graph.neighbors(src_id):
                if nb in id_to_idx:
                    edge_src.append(id_to_idx[src_id])
                    edge_dst.append(id_to_idx[nb])

        # Collect real per-node embeddings from SQLite so the GNN's input features
        # actually encode node identity/content, not just graph structure. Nodes with
        # no stored embedding (e.g. pure-graph-expansion candidates) fall back to a
        # zero vector — the GraphSAGE aggregation still lets their neighbors' real
        # features propagate in via message passing.
        feat_dim = len(query_embedding) if hasattr(query_embedding, "__len__") else 1024
        x = torch.zeros(len(node_ids), feat_dim, dtype=torch.float32)
        if sqlite_store is not None:
            for i, nid in enumerate(node_ids):
                try:
                    node = sqlite_store.get_node(nid)
                except Exception:
                    node = None
                if node is not None:
                    emb = node.get("embedding")
                    if emb is not None and len(emb) == feat_dim:
                        x[i] = torch.tensor(np.asarray(emb, dtype=np.float32))

        edge_index = torch.tensor(
            [edge_src, edge_dst], dtype=torch.long
        ) if edge_src else torch.zeros((2, 0), dtype=torch.long)

        device = self._device or "cpu"
        x = x.to(device)
        edge_index = edge_index.to(device)

        with torch.no_grad():
            gnn_embeddings = self._model(x, edge_index).cpu().numpy()

        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)

        # combined_score (RRF, ~1e-2 magnitude) and gnn_score (cosine, [-1,1]) are on
        # very different scales — min-max normalize combined_score across this
        # candidate pool before blending so neither signal is drowned out.
        raw_combined = [cand.get("combined_score", 0.0) for cand in candidates]
        lo, hi = min(raw_combined, default=0.0), max(raw_combined, default=0.0)
        span = (hi - lo) or 1.0

        scores = []
        for i, cand in enumerate(candidates):
            emb = gnn_embeddings[i]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            gnn_score = float(np.dot(q_norm, emb_norm))
            norm_combined = (raw_combined[i] - lo) / span
            blended = 0.5 * norm_combined + 0.5 * gnn_score
            scores.append((blended, cand))

        scores.sort(key=lambda x: -x[0])
        result = [c for _, c in scores]
        return result[:top_k] if top_k else result


_gnn_reranker: Optional[GNNReranker] = None


def get_gnn_reranker() -> Optional[GNNReranker]:
    global _gnn_reranker
    try:
        from config import settings
        if not settings.gnn_enabled:
            return None
    except Exception:
        return None

    if _gnn_reranker is None:
        try:
            from config import settings
            ckpt = getattr(settings, "gnn_model_path", None)
        except Exception:
            ckpt = None
        _gnn_reranker = GNNReranker(checkpoint_path=ckpt)

    return _gnn_reranker if _gnn_reranker.is_available() else None
