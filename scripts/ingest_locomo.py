"""Ingest LoCoMo benchmark data into HybridMind for retrieval eval."""
import json
import sys
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

LOCOMO_PATH = Path("memorybench/data/benchmarks/locomo/locomo10.json")
BASE_URL = "http://127.0.0.1:8000"

def main():
    client = httpx.Client(timeout=60)

    try:
        resp = client.get(f"{BASE_URL}/health")
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
                resp = client.post(f"{BASE_URL}/nodes", json={
                    "text": text_with_meta,
                    "metadata": {
                        "session_id": sample_id,
                        "sessionId": sample_id,
                        "containerTag": "locomo",
                        "role": role,
                        "timestamp": date_str,
                    }
                })
                if resp.status_code == 201:
                    total_nodes += 1
            except Exception as e:
                print(f"  ERROR creating node: {e}")

        # Ingest facts via session-facts endpoint
        try:
            resp = client.post(f"{BASE_URL}/ingest/session-facts", json={
                "session_id": sample_id,
                "turns": all_turns,
                "container_tag": "locomo",
            })
            if resp.status_code == 200:
                fj = resp.json()
                nf = fj.get("facts_extracted", 0)
                total_nodes += nf
                print(f"  [{i+1}/{len(data)}] session {sample_id}: {len(all_turns)} turns, {nf} facts")
            else:
                print(f"  [{i+1}/{len(data)}] WARN: session-facts returned {resp.status_code}")
        except Exception as e:
            print(f"  [{i+1}/{len(data)}] ERROR session-facts: {e}")

    print(f"\nIngestion complete: {total_nodes} nodes across {len(data)} sessions")
    print("Ready: python eval_locomo_retrieval.py")

if __name__ == "__main__":
    main()
