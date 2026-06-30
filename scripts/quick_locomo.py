"""Quick LoCoMo subset eval — ingest a few conversations and measure retrieval."""
import json, sys, time
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL = "http://127.0.0.1:8000"
LOCOMO_PATH = Path("memorybench/data/benchmarks/locomo/locomo10.json")

def main():
    client = httpx.Client(timeout=60)
    
    # Health check
    r = client.get(f"{BASE_URL}/health")
    print(f"Server: {r.json()['status']}")
    
    # Clear first
    client.delete(f"{BASE_URL}/admin/clear")
    
    data = json.loads(LOCOMO_PATH.read_text())
    conv = data[0]
    convo = conv["conversation"]
    
    # Collect first 20 turns from session 1
    all_turns = []
    for key in sorted(convo.keys()):
        if key.endswith("_date_time"):
            continue
        if not key.startswith("session_"):
            continue
        date_key = key + "_date_time"
        date = convo.get(date_key, "")
        msgs = convo[key]
        if not isinstance(msgs, list):
            continue
        for msg in msgs:
            if isinstance(msg, dict) and "text" in msg:
                spk = msg.get("speaker", "unknown")
                txt = msg.get("text", "").strip()
                if txt:
                    all_turns.append({"speaker": spk, "text": txt, "date": str(date)})
    print(f"Total turns available: {len(all_turns)}")
    
    # Ingest just session 1 (~30 turns)
    session_1_key = next(k for k in sorted(convo.keys()) if k == "session_1")
    date = convo.get("session_1_date_time", "")
    msgs = convo[session_1_key]
    print(f"Session 1: {len(msgs)} messages, date={date}")
    
    sample_id = conv.get("sample_id", "locomo_0")
    
    for msg in msgs[:20]:
        if not isinstance(msg, dict) or "text" not in msg:
            continue
        spk = msg.get("speaker", "unknown")
        txt = msg["text"].strip()
        r = client.post(f"{BASE_URL}/nodes", json={
            "text": f"[DATE: {date}] [SPEAKER: {spk}] {txt}" if date else f"[SPEAKER: {spk}] {txt}",
            "metadata": {
                "session_id": sample_id,
                "sessionId": sample_id,
                "containerTag": "locomo",
                "role": "ai" if spk.startswith("speaker_b") or spk.startswith("speaker_a")[:2] == "sp" else "human",
                "timestamp": str(date),
            }
        })
        if r.status_code != 201:
            print(f"WARN: {r.status_code}")
    
    # Get stats
    r = client.get(f"{BASE_URL}/search/stats")
    stats = r.json()
    print(f"Ingested. Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")
    
    # Run a few LoCoMo queries
    qas = conv.get("qa", [])[:3]  # first 3 questions
    for qa in qas:
        q = qa["question"]
        ans = qa.get("answer") or qa.get("adversarial_answer", "")
        cat = {1: "single-hop", 2: "temporal", 3: "multi-hop"}.get(qa.get("category", 0), "?")
        print(f"\n[{cat}] Q: {q}")
        print(f"  Gold answer: {str(ans)[:120]}")
        
        # Search with both RRF and linear
        for mode in ["rrf", "linear"]:
            r = client.post(f"{BASE_URL}/search/hybrid", json={
                "query_text": q,
                "top_k": 5,
                "vector_weight": 0.5,
                "graph_weight": 0.15,
                "bm25_boost_weight": 0.35,
                "rerank_pool": 25,
                "fusion_mode": mode,
                "filter_metadata": {"containerTag": "locomo"},
            })
            results = r.json().get("results", [])
            for rank, res in enumerate(results[:3], 1):
                text = res.get("text", "")[:100]
                cs = res.get("combined_score", 0)
                rs = res.get("rerank_score", "-")
                print(f"  [{mode}] #{rank} c={cs:.4f} rerank={rs} {text}...")
    
if __name__ == "__main__":
    main()
