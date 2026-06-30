"""
Train the GNN reranker for HybridMind (intended for RunPod GPU).

Architecture: 2-layer GraphSAGE over the HybridMind typed graph.
Loss: pairwise BPR (Bayesian Personalised Ranking) — positive nodes (ground-truth
      for query) ranked above negative nodes from the same candidate pool.

Usage:
    python scripts/train_gnn.py \
        --mind-path data/hybridmind.mind \
        --data-path benchmarks/fusion_train_data.jsonl \
        --output models/gnn.pt \
        --epochs 50

Requires: pip install torch-geometric

The output checkpoint is loadable by engine/gnn_reranker.py.
Set HYBRIDMIND_GNN_ENABLED=true and HYBRIDMIND_GNN_MODEL_PATH=<path> to use it.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_gnn")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mind-path", default="data/hybridmind.mind")
    parser.add_argument("--data-path", default="benchmarks/fusion_train_data.jsonl")
    parser.add_argument("--output", default="models/gnn.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv
        from torch_geometric.data import Data
    except ImportError:
        logger.error("torch-geometric is required. Install with: pip install torch-geometric")
        sys.exit(1)

    from config import settings
    from storage.sqlite_store import SQLiteStore
    from storage.graph_index import GraphIndex
    from storage.mindfile import MindFile
    from engine.embedding import EmbeddingEngine
    from engine.device import resolve_device

    device = resolve_device(settings.device)
    logger.info(f"Device: {device}")

    mind_file = MindFile(args.mind_path)
    paths = mind_file.get_paths()
    sqlite_store = SQLiteStore(paths["sqlite"])
    graph_index = GraphIndex(index_path=paths["graph"])

    embedding_engine = EmbeddingEngine(model_name=settings.embedding_model, device=device)

    logger.info("Loading nodes and building graph tensors...")
    all_nodes = sqlite_store.list_nodes(limit=10_000_000)
    node_ids = [n["id"] for n in all_nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    logger.info(f"Embedding {len(node_ids)} nodes...")
    texts = [n["text"] for n in all_nodes]
    embeddings = embedding_engine.embed_batch(texts, normalize=True, batch_size=32, show_progress=True)
    import numpy as np
    x = torch.tensor(embeddings, dtype=torch.float32)
    in_dim = x.shape[1]

    logger.info("Building edge index...")
    edge_src, edge_dst = [], []
    for src_id in node_ids:
        if not graph_index.has_node(src_id):
            continue
        for nb in graph_index.graph.neighbors(src_id):
            if nb in id_to_idx:
                edge_src.append(id_to_idx[src_id])
                edge_dst.append(id_to_idx[nb])
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index).to(device)

    logger.info("Loading training data...")
    records = []
    with open(args.data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} queries")

    class GraphSAGE(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
            super().__init__()
            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, out_dim))

        def forward(self, x, edge_index):
            for conv in self.convs[:-1]:
                x = F.relu(conv(x, edge_index))
            return self.convs[-1](x, edge_index)

    model = GraphSAGE(in_dim, args.hidden, args.hidden, args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        node_embs = model(data.x, data.edge_index)
        total_loss = 0.0
        n_pairs = 0

        for rec in records:
            cands = rec.get("candidates", [])
            positives = [c["node_id"] for c in cands if c.get("is_relevant") and c["node_id"] in id_to_idx]
            negatives = [c["node_id"] for c in cands if not c.get("is_relevant") and c["node_id"] in id_to_idx]
            if not positives or not negatives:
                continue

            pos_embs = node_embs[[id_to_idx[nid] for nid in positives]].mean(dim=0)
            neg_embs = node_embs[[id_to_idx[nid] for nid in negatives]].mean(dim=0)
            loss = -torch.log(torch.sigmoid(pos_embs.dot(pos_embs) - neg_embs.dot(neg_embs)) + 1e-8)
            total_loss += loss.item()
            n_pairs += 1
            loss.backward(retain_graph=True)

        optimizer.step()
        optimizer.zero_grad()
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}  loss={total_loss/max(1,n_pairs):.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_dim": in_dim,
        "hidden_dim": args.hidden,
        "out_dim": args.hidden,
        "num_layers": args.layers,
    }, args.output)
    logger.info(f"GNN checkpoint saved to {args.output}")
    logger.info(
        "Set HYBRIDMIND_GNN_ENABLED=true and HYBRIDMIND_GNN_MODEL_PATH=<path> to activate."
    )


if __name__ == "__main__":
    main()
