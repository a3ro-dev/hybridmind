"""Ingest LoCoMo benchmark data into HybridMind for retrieval eval.

Uses asyncio + a semaphore to fire up to CONCURRENCY parallel POST /nodes
requests instead of one-at-a-time. ~16x faster than the sequential version.
"""
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

LOCOMO_PATH = Path("memorybench/data/benchmarks/locomo/locomo10.json")
BASE_URL = os.getenv("HYBRIDMIND_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
CONCURRENCY = 16   # parallel /nodes requests
TIMEOUT = 300.0


async def post_node(client: httpx.AsyncClient, sem: asyncio.Semaphore, payload: dict) -> bool:
    """POST one node with retries under the semaphore."""
    async with sem:
        for attempt in range(6):
            try:
                resp = await client.post("/nodes", json=payload)
                if resp.status_code in (200, 201):
                    return True
                print(f"  [attempt {attempt+1}] HTTP {resp.status_code}: {resp.text[:80]}", flush=True)
            except Exception as e:
                print(f"  [attempt {attempt+1}] {type(e).__name__}: {e}", flush=True)
            await asyncio.sleep(1.5 * (attempt + 1))
        return False


async def ingest_session(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    sample_id: str,
    all_turns: list,
    speakers: dict,
    idx: int,
    total: int,
) -> int:
    """Ingest all turns for one conversation concurrently, then run fact extraction."""
    tasks = []
    for turn in all_turns:
        role = "ai" if turn["speaker"].startswith(speakers.get("speaker_b", "ai")[:3]) else "human"
        date_str = turn["date"]
        txt = turn["text"]
        text_with_meta = (
            f"[DATE: {date_str}] [SPEAKER: {turn['speaker']}] {txt}"
            if date_str else
            f"[SPEAKER: {turn['speaker']}] {txt}"
        )
        payload = {
            "text": text_with_meta,
            "metadata": {
                "session_id": sample_id,
                "sessionId": sample_id,
                "containerTag": "locomo",
                "role": role,
                "timestamp": date_str,
            },
        }
        tasks.append(post_node(client, sem, payload))

    results = await asyncio.gather(*tasks)
    ok = sum(results)
    failed = len(results) - ok
    print(f"  [{idx}/{total}] {sample_id}: {ok}/{len(all_turns)} turns ingested"
          + (f" ({failed} failed)" if failed else ""), flush=True)

    # Fact extraction (LLM) — fire and forget errors, non-blocking
    try:
        resp = await client.post("/ingest/session-facts", json={
            "session_id": sample_id,
            "turns": all_turns,
            "container_tag": "locomo",
        })
        fj = resp.json()
        nf = fj.get("facts_extracted", 0)
        print(f"  [{idx}/{total}] {sample_id}: {nf} facts extracted", flush=True)
        ok += nf
    except Exception as e:
        print(f"  [{idx}/{total}] {sample_id}: fact extraction error: {e}", flush=True)

    return ok


async def main():
    # Health check
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as c:
        try:
            # /health performs a real remote embedding and can block the single
            # API worker during a serverless cold start. Readiness only verifies
            # that the initialized engine and indexes are available.
            r = await c.get("/ready")
            r.raise_for_status()
            print("HybridMind is healthy", flush=True)
        except Exception as e:
            print(f"HybridMind not reachable: {e}")
            sys.exit(1)

    data = json.loads(LOCOMO_PATH.read_text())
    print(f"Loaded {len(data)} conversations", flush=True)

    sem = asyncio.Semaphore(CONCURRENCY)
    total_nodes = 0

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=TIMEOUT,
        limits=httpx.Limits(max_connections=CONCURRENCY + 4, max_keepalive_connections=CONCURRENCY),
    ) as client:
        for i, conv in enumerate(data, 1):
            sample_id = conv.get("sample_id", f"locomo_{i}")

            # Skip already-ingested sessions
            try:
                with sqlite3.connect("data/hybridmind.mind/store.db") as conn:
                    existing = conn.execute(
                        "SELECT COUNT(*) FROM nodes WHERE "
                        "json_extract(metadata, '$.session_id') = ? OR "
                        "json_extract(metadata, '$.sessionId') = ?",
                        (sample_id, sample_id),
                    ).fetchone()[0]
                if existing > 0:
                    print(f"  [{i}/{len(data)}] {sample_id}: already ingested ({existing} nodes), skipping", flush=True)
                    total_nodes += existing
                    continue
            except Exception:
                pass  # DB doesn't exist yet — first run

            convo = conv.get("conversation", {})
            speakers = {
                "speaker_a": convo.get("speaker_a", "human"),
                "speaker_b": convo.get("speaker_b", "ai"),
            }

            all_turns = []
            for key in sorted(convo.keys()):
                if key.startswith("session_") and not key.endswith("_date_time"):
                    date = convo.get(key + "_date_time", "")
                    messages = convo[key]
                    if not isinstance(messages, list):
                        continue
                    for msg in messages:
                        if isinstance(msg, dict) and "text" in msg:
                            txt = msg["text"].strip()
                            if txt:
                                all_turns.append({
                                    "speaker": msg.get("speaker", "unknown"),
                                    "text": txt,
                                    "date": date,
                                })

            if not all_turns:
                print(f"  [{i}/{len(data)}] {sample_id}: no turns, skipping", flush=True)
                continue

            n = await ingest_session(client, sem, sample_id, all_turns, speakers, i, len(data))
            total_nodes += n

    print(f"\nDone: {total_nodes} nodes across {len(data)} sessions", flush=True)
    print("Ready: python eval_locomo_retrieval.py")


if __name__ == "__main__":
    asyncio.run(main())
