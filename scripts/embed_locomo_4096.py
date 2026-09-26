"""Embed LoCoMo turns and questions with the production native-4096 TEI endpoint (plan-bound).

Documents use the sparse `spk_cap` text (speaker + text + image caption). Each question is
embedded twice: bare (what production sends today) and with Qwen3-Embedding's documented
query instruction ("Instruct: {task}\\nQuery:{query}"), so the missing instruction can be
measured. Vectors go to a local .npz cache (not committed); a JSON receipt records calls,
wall time, status and the cache SHA-256. Stops at the plan ceiling; failures are receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.offline_budgeted_evidence import DATASET, index_text, load  # noqa: E402

INSTRUCTION = "Given a question about a long conversation, retrieve the conversation turns that answer it"


def instructed(question: str) -> str:
    return f"Instruct: {INSTRUCTION}\nQuery:{question}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help=".npz vector cache")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    from engine.embedding import get_embedding_engine
    from engine.resource_accounting import load_and_validate_live_plan

    plan, gate = load_and_validate_live_plan(args.plan)
    if "tei" not in plan["providers"]:
        raise SystemExit("plan does not admit the TEI embedding provider")
    ceiling = plan["usage_ceiling"]
    convs = load(DATASET)
    doc_ids = [f"{c['sid']}:{t['id']}" for c in convs for t in c["turns"]]
    docs = [index_text(t, "spk_cap") for c in convs for t in c["turns"]]
    q_ids = [q["qid"] for c in convs for q in c["qs"]]
    questions = [q["q"] for c in convs for q in c["qs"]]
    texts = docs + questions + [instructed(q) for q in questions]
    calls_needed = 1 + -(-len(texts) // args.batch_size)  # warmup + batches
    if calls_needed > ceiling["embedding_calls"]:
        raise SystemExit(f"needs {calls_needed} calls > ceiling {ceiling['embedding_calls']}")

    receipt = {"schema": "hybridmind.embed_locomo_4096.v1", "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
               "gate": vars(gate), "texts": len(texts), "instruction": INSTRUCTION, "calls": 0, "status": "started",
               "started_at": datetime.now(timezone.utc).isoformat()}
    engine = get_embedding_engine()
    t0 = time.monotonic()
    vectors = []
    try:
        engine.warmup(timeout_s=180.0)
        receipt["calls"] += 1
        receipt["warmup_seconds"] = time.monotonic() - t0
        for i in range(0, len(texts), args.batch_size):
            if time.monotonic() - t0 > ceiling["provider_runtime_seconds"]:
                raise RuntimeError("provider runtime ceiling reached")
            vectors.append(engine.embed_batch(texts[i:i + args.batch_size], batch_size=args.batch_size))
            receipt["calls"] += 1
        matrix = np.vstack(vectors).astype(np.float32)
        if matrix.shape != (len(texts), 4096) or not np.isfinite(matrix).all():
            raise RuntimeError(f"invalid embedding matrix {matrix.shape}")
        n_d, n_q = len(docs), len(questions)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, doc_ids=np.array(doc_ids), q_ids=np.array(q_ids), docs=matrix[:n_d],
                 q_bare=matrix[n_d:n_d + n_q], q_inst=matrix[n_d + n_q:])
        receipt.update(status="completed", cache_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    except Exception as exc:  # fail closed: the receipt is the evidence
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:400])
    finally:
        receipt.update(wall_seconds=time.monotonic() - t0, finished_at=datetime.now(timezone.utc).isoformat())
        args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
        print(json.dumps({k: receipt[k] for k in ("status", "calls", "wall_seconds", "texts")}))
    return 0 if receipt["status"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
