"""
Re-embed all nodes and rebuild the FAISS vector index.

Run this when migrating an existing corpus to HybridMind's mandatory remote
4096-dimensional embedding contract or when changing the remote model while
preserving that exact dimension.

Usage:
    python scripts/reindex_embeddings.py [--batch-size 32] [--dry-run]

The script:
1. Reads all node texts from SQLite.
2. Re-embeds them with the currently configured model.
3. Writes the new embeddings back to SQLite (overwrites existing).
4. Rebuilds the FAISS index from scratch and saves it to .mind.
5. Updates the .mind manifest with the new model and dimension.
"""
import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reindex")


def main():
    parser = argparse.ArgumentParser(description="Re-embed all nodes and rebuild FAISS index")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--dry-run", action="store_true", help="Only print counts, do not write")
    args = parser.parse_args()

    from config import settings
    from storage.sqlite_store import SQLiteStore
    from storage.vector_index import VectorIndex
    from storage.mindfile import MindFile
    from engine.device import resolve_device

    logger.info(f"Model:     {settings.embedding_model}")
    logger.info(f"Dimension: {settings.embedding_dimension}")
    logger.info(f"Batch:     {args.batch_size}")
    logger.info(f"Dry run:   {args.dry_run}")

    device = resolve_device(settings.device)
    logger.info(f"Device:    {device}")

    mind_file = MindFile(settings.mind_file_path)
    if not mind_file.exists:
        logger.error(f".mind file not found at {settings.mind_file_path}. Nothing to reindex.")
        sys.exit(1)

    paths = mind_file.get_paths()
    sqlite_store = SQLiteStore(paths["sqlite"])

    all_nodes = sqlite_store.list_nodes(limit=10_000_000)
    total = len(all_nodes)
    logger.info(f"Found {total} nodes to re-embed")

    if total == 0:
        logger.info("Nothing to do.")
        return

    if args.dry_run:
        logger.info("Dry run — exiting without changes.")
        return

    from engine.embedding import get_embedding_engine
    embedding_engine = get_embedding_engine(
        model_name=settings.embedding_model,
        device=device,
    )

    # Do NOT pass index_path here — we don't want to load the old (wrong-dim) index.
    # After rebuild we'll save to the correct path manually.
    vector_index = VectorIndex(
        dimension=settings.embedding_dimension,
        index_path=None,
        hnsw_ef_search=settings.hnsw_ef_search,
        hnsw_ef_construction=settings.hnsw_ef_construction,
    )
    vector_index.index_path = Path(paths["vector_index"])

    texts = [n["text"] for n in all_nodes]
    node_ids = [n["id"] for n in all_nodes]

    logger.info("Generating embeddings...")
    t0 = time.perf_counter()
    embeddings = embedding_engine.embed_batch(
        texts,
    )
    expected_shape = (total, 4096)
    if embeddings.shape != expected_shape:
        raise RuntimeError(
            f"Embedding backend returned shape {embeddings.shape}; reindexing requires "
            f"exactly {expected_shape} and will not accept missing/extra rows."
        )
    if not np.all(np.isfinite(embeddings)):
        raise RuntimeError("Embedding backend returned non-finite values; refusing reindex")
    elapsed = time.perf_counter() - t0
    logger.info(f"Embedding done: {total} nodes in {elapsed:.1f}s ({total/elapsed:.1f} nodes/s)")
    logger.info(f"Embedding shape: {embeddings.shape}")

    logger.info("Writing embeddings to SQLite...")
    for node_id, emb in zip(node_ids, embeddings):
        sqlite_store.update_node(node_id, embedding=emb, raw_embedding=emb)

    logger.info("Rebuilding FAISS index...")
    pairs = list(zip(node_ids, embeddings))
    vector_index.rebuild_from_embeddings(pairs)

    logger.info("Saving FAISS index to disk...")
    vector_index.save()

    logger.info("Updating .mind manifest...")
    mind_file.initialize(metadata={
        "description": "HybridMind Vector + Graph Database",
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
    })

    logger.info(f"Reindex complete. {total} nodes at {settings.embedding_dimension}-dim.")


if __name__ == "__main__":
    main()
