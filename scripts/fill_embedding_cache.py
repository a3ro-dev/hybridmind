"""Fill the 4096-d embedding cache that ``eval_trisignal.py --dense cache:PATH`` replays.

Dry run (default, zero provider calls): counts the documents and queries a
dataset needs, how many are already cached, and a whitespace-token estimate.

Live (``--execute --plan PLAN``): validates the priced live plan with the same
gate as other live runs (``engine.resource_accounting``), requires the plan to
admit the ``tei`` provider, refuses to start if the missing items need more
embedding calls than the plan's ``usage_ceiling.embedding_calls``, then embeds
only the missing items through the configured embedding engine (for a VPS:
``LOCAL_TEI_EMBEDDING_URL=http://127.0.0.1:8080`` serving Qwen3-Embedding-8B).

Documents are embedded bare (``TurnRecord.search_text``); queries are embedded
with ``--query-instruction`` in the Qwen3 format when one is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks import conversational_data as data  # noqa: E402
from engine.embedding_cache import CacheItem, EmbeddingCache  # noqa: E402

DEFAULT_PATHS = {
    "locomo": ROOT / "memorybench/data/benchmarks/locomo/locomo10.json",
    "longmemeval": ROOT / "memorybench/data/benchmarks/longmemeval/longmemeval_s_cleaned.json",
}


def dataset_items(dataset: str, path: Path, query_instruction: str, limit=None) -> List[CacheItem]:
    """Unique doc and query items in first-seen order (shared sessions dedupe)."""
    convs = data.load_locomo(path) if dataset == "locomo" else data.load_longmemeval(path, limit=limit)
    seen, items = set(), []
    for conv in convs:
        batch = [CacheItem("doc", t.search_text) for t in conv.corpus.turns]
        batch += [CacheItem("query", q.question, query_instruction) for q in conv.questions]
        for item in batch:
            if item not in seen:
                seen.add(item)
                items.append(item)
    return items


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DEFAULT_PATHS), required=True)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--query-instruction", default="")
    # TEI's default --max-client-batch-size is 32; one batch is one HTTP call.
    parser.add_argument("--batch-size", type=int, default=32, choices=range(1, 33), metavar="1..32")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args(argv)

    path = args.path or DEFAULT_PATHS[args.dataset]
    identity = data.dataset_identity(path)
    items = dataset_items(args.dataset, path, args.query_instruction, args.limit)
    with EmbeddingCache(args.cache, model_id=args.model, expected_dim=4096) as cache:
        _, missing = cache.get_many(items)
        calls = -(-len(missing) // args.batch_size)
        report = {
            "dataset": identity, "model": args.model, "query_instruction": args.query_instruction,
            "items": len(items), "cached": len(items) - len(missing), "missing": len(missing),
            "missing_whitespace_tokens": sum(len(i.text.split()) for i in missing),
            "embedding_calls_needed": calls, "batch_size": args.batch_size,
        }
        print(json.dumps(report, indent=2, default=str))
        if not args.execute or not missing:
            return 0
        if args.plan is None:
            parser.error("--execute requires --plan (a priced, preflight-validated live plan)")

        from engine.resource_accounting import load_and_validate_live_plan

        plan, _gate = load_and_validate_live_plan(args.plan)
        if "tei" not in plan.get("providers", []):
            raise SystemExit("plan does not admit the tei provider")
        ceiling = int(plan.get("usage_ceiling", {}).get("embedding_calls", 0))
        if calls > ceiling:
            raise SystemExit(f"{calls} embedding calls needed but the plan allows {ceiling}")

        from engine.embedding import format_query_for_embedding, get_embedding_engine

        from engine.embedding import TEIEmbeddingEngine

        engine = get_embedding_engine()
        if not isinstance(engine, TEIEmbeddingEngine):
            raise SystemExit("the configured embedding backend is not TEI; the plan admits only tei")
        for start in range(0, len(missing), args.batch_size):
            batch = missing[start:start + args.batch_size]
            texts = [
                format_query_for_embedding(i.text, i.instruction, "qwen3") if i.kind == "query" else i.text
                for i in batch
            ]
            cache.put_many(batch, engine.embed_batch(texts, batch_size=args.batch_size))
            print(f"embedded {start + len(batch)}/{len(missing)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
