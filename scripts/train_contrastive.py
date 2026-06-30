"""
Contrastive fine-tuning of bge-m3 on stored HybridMind memory pairs.

Approach: SimCSE-style unsupervised contrastive learning.
- Positives: node pairs connected by strong graph edges (led_to, caused_by,
  supports, derived_from, depends_on) — these encode semantic proximity.
- Negatives: in-batch random negatives (standard SimCSE).

This replaces the ~1% neighbourhood-averaging effect with a proper
graph-aware embedding space, trained end-to-end.

Output: a bge-m3 checkpoint fine-tuned on your graph's edge structure,
loadable via HYBRIDMIND_EMBEDDING_MODEL=<output_dir>.

Usage (RunPod):
    python scripts/train_contrastive.py \
        --mind-path data/hybridmind.mind \
        --output models/bge-m3-graph-finetuned \
        --epochs 3 \
        --batch-size 32 \
        --min-edge-weight 0.5

Requirements:
    pip install FlagEmbedding>=1.2.10 transformers datasets

Note:
- This script is designed for RunPod (GPU). On CPU it will run but very slowly.
- Set HYBRIDMIND_EMBEDDING_MODEL=<output_dir> to use the fine-tuned checkpoint.
- The base bge-m3 model is not modified in-place; a new checkpoint is written.
"""
import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_contrastive")

STRONG_EDGE_TYPES = {"led_to", "caused_by", "supports", "derived_from", "depends_on", "similar_to"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mind-path", default="data/hybridmind.mind")
    parser.add_argument("--output", default="models/bge-m3-graph-finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-pairs", type=int, default=50_000,
                        help="Maximum edge pairs to use for training")
    parser.add_argument("--min-edge-weight", type=float, default=0.5,
                        help="Only use edges above this weight")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        logger.error("PyTorch is required. Install: pip install torch")
        sys.exit(1)

    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        logger.error("transformers is required. Install: pip install transformers")
        sys.exit(1)

    from config import settings
    from storage.sqlite_store import SQLiteStore
    from storage.graph_index import GraphIndex
    from storage.mindfile import MindFile
    from engine.device import resolve_device

    device_str = resolve_device(settings.device)
    device = torch.device(device_str)
    logger.info(f"Device: {device}")

    # Load graph and find strong edge pairs
    mind_file = MindFile(args.mind_path)
    if not mind_file.exists:
        logger.error(f".mind file not found at {args.mind_path}")
        sys.exit(1)

    paths = mind_file.get_paths()
    sqlite_store = SQLiteStore(paths["sqlite"])
    graph_index = GraphIndex(index_path=paths["graph"])

    logger.info("Collecting training pairs from graph edges...")
    pairs = []
    for u, v, data in graph_index.graph.edges(data=True):
        edge_type = data.get("type", "")
        weight = float(data.get("weight", 1.0))
        if edge_type in STRONG_EDGE_TYPES and weight >= args.min_edge_weight:
            pairs.append((u, v))

    logger.info(f"Found {len(pairs)} strong edge pairs (before sampling)")
    if not pairs:
        logger.error(
            "No strong edges found. Ingest data with edges of types: "
            + ", ".join(STRONG_EDGE_TYPES)
        )
        sys.exit(1)

    random.shuffle(pairs)
    pairs = pairs[: args.max_pairs]
    logger.info(f"Using {len(pairs)} pairs for training")

    # Fetch node texts
    logger.info("Fetching node texts...")
    node_texts = {}
    all_node_ids = list({nid for pair in pairs for nid in pair})
    for nid in all_node_ids:
        node = sqlite_store.get_node(nid)
        if node:
            node_texts[nid] = node["text"]

    valid_pairs = [(u, v) for u, v in pairs if u in node_texts and v in node_texts]
    logger.info(f"Valid pairs with texts: {len(valid_pairs)}")

    if len(valid_pairs) < args.batch_size:
        logger.error(f"Need at least {args.batch_size} valid pairs; got {len(valid_pairs)}")
        sys.exit(1)

    # Load bge-m3 for fine-tuning (dense encoder head only)
    model_name = settings.embedding_model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def mean_pool(hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def encode(texts):
        enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        return F.normalize(emb, dim=-1)

    # SimCSE contrastive loss
    def nt_xent_loss(q, k, temperature=0.05):
        sim = torch.matmul(q, k.T) / temperature  # (B, B)
        labels = torch.arange(sim.size(0), device=device)
        return F.cross_entropy(sim, labels)

    logger.info(f"Training for {args.epochs} epochs, batch_size={args.batch_size}")
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(valid_pairs)
        total_loss = 0.0
        steps = 0
        for i in range(0, len(valid_pairs) - args.batch_size, args.batch_size):
            batch = valid_pairs[i : i + args.batch_size]
            anchors = [node_texts[u] for u, _ in batch]
            positives = [node_texts[v] for _, v in batch]

            q_emb = encode(anchors)
            k_emb = encode(positives)
            loss = nt_xent_loss(q_emb, k_emb, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg = total_loss / max(1, steps)
        logger.info(f"Epoch {epoch+1}/{args.epochs}  loss={avg:.4f}")

    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    logger.info(f"Fine-tuned model saved to {output_path}")
    logger.info(
        f"To use: HYBRIDMIND_EMBEDDING_MODEL={output_path}  HYBRIDMIND_EMBEDDING_DIMENSION=1024"
    )
    logger.info("Then run: python scripts/reindex_embeddings.py")


if __name__ == "__main__":
    main()
