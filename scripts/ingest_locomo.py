"""Ingest LoCoMo benchmark data into HybridMind for retrieval eval."""
import json
import sys
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

LOCOMO_PATH = Path("memorybench/data/benchmarks/locomo/locomo10.json")
BASE_URL = "http://127.0.0.1:8000"

import time

def post_with_retry(url: str, json_data: dict, max_retries: int = 5) -> httpx.Response:
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=300.0) as fresh_client:
                resp = fresh_client.post(url, json=json_data)
                if resp.status_code in (200, 201):
                    return resp
                print(f"  [Attempt {attempt+1}] HTTP {resp.status_code}: {resp.text[:100]}", flush=True)
        except Exception as e:
            print(f"  [Attempt {attempt+1}] Connection error: {e}", flush=True)
        time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Failed POST to {url} after {max_retries} retries")

def main():
    try:
        with httpx.Client(timeout=30.0) as c:
            resp = c.get(f"{BASE_URL}/health")
            resp.raise_for_status()
            print("HybridMind is healthy")
    except Exception as e:
        print(f"HybridMind not reachable: {e}")
        sys.exit(1)

    data = json.loads(LOCOMO_PATH.read_text())
    print(f"Loaded {len(data)} conversations")

    total_nodes = 0

    for i, conv in enumerate(data):
        sample_id = conv.get("sample_id", f"locomo_{i}")
        import sqlite3
        with sqlite3.connect('data/hybridmind.mind/store.db') as conn:
            existing = conn.execute("SELECT COUNT(*) FROM nodes WHERE json_extract(metadata, '$.sessionId') = ? OR json_extract(metadata, '$.session_id') = ?", (sample_id, sample_id)).fetchone()[0]
        if existing > 0:
            print(f"  [{i+1}/{len(data)}] session {sample_id}: already ingested ({existing} nodes), skipping...", flush=True)
            total_nodes += existing
            continue

        convo = conv.get("conversation", {})

        # Collect all turns across all sessions
        all_turns = []
        speakers = {"speaker_a": convo.get("speaker_a", "human"),
                     "speaker_b": convo.get("speaker_b", "ai")}

        for key in sorted(convo.keys()):
            if key.startswith("session_") and not key.endswith("_date_time"):
                date_key = key + "_date_time"
                date = convo.get(date_key, "")
                messages = convo[key]
                if not isinstance(messages, list):
                    continue
                for msg in messages:
                    if isinstance(msg, dict) and "text" in msg:
                        spk = msg.get("speaker", "unknown")
                        txt = msg["text"].strip()
                        if txt:
                            all_turns.append({
                                "speaker": spk, "text": txt, "date": date
                            })

        if not all_turns:
            print(f"  [{i+1}/{len(data)}] session {sample_id}: no turns found, skipping")
            continue

        # Ingest turns one at a time
        for turn in all_turns:
            role = "ai" if turn["speaker"].startswith(speakers.get("speaker_b", "ai")[:3]) else "human"
            date_str = turn["date"]
            txt = turn["text"]
            text_with_meta = f"[DATE: {date_str}] [SPEAKER: {turn['speaker']}] {txt}" if date_str else f"[SPEAKER: {turn['speaker']}] {txt}"
            try:
                resp = post_with_retry(f"{BASE_URL}/nodes", {
                    "text": text_with_meta,
                    "metadata": {
                        "session_id": sample_id,
                        "sessionId": sample_id,
                        "containerTag": "locomo",
                        "role": role,
                        "timestamp": date_str,
                    }
                })
                total_nodes += 1
            except Exception as e:
                print(f"  ERROR creating node: {e}")

        print(f"  [{i+1}/{len(data)}] {sample_id}: ingested {len(all_turns)} turn nodes (total={total_nodes})", flush=True)

        # Ingest facts via session-facts endpoint
        try:
            resp = post_with_retry(f"{BASE_URL}/ingest/session-facts", {
                "session_id": sample_id,
                "turns": all_turns,
                "container_tag": "locomo",
            })
            fj = resp.json()
            nf = fj.get("facts_extracted", 0)
            total_nodes += nf
            print(f"  [{i+1}/{len(data)}] session {sample_id}: {len(all_turns)} turns, {nf} facts", flush=True)
        except Exception as e:
            print(f"  ERROR session-facts: {e}", flush=True)

    print(f"\nIngestion complete: {total_nodes} nodes across {len(data)} sessions", flush=True)
    print("Ready: python eval_locomo_retrieval.py")

if __name__ == "__main__":
    main()
