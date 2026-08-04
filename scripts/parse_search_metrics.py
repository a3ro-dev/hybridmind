import json

def main():
    print("Loading checkpoint...")
    with open('/home/azureuser/hybridmind/memorybench/data/runs/hybridmind-locomo-fixed-20260726/checkpoint.json', 'r') as f:
        cp = json.load(f)

    qs = cp.get('questions', {})
    total_ingested = len(qs)
    searched = [q for q in qs.values() if q.get('phases', {}).get('search', {}).get('status') == 'completed']

    conv_counts = {}
    latencies = []
    context_counts = []
    recall_hits = 0
    total_with_evidence = 0

    for q in searched:
        c_id = q.get('conversationId', 'unknown')
        conv_counts[c_id] = conv_counts.get(c_id, 0) + 1
        
        s_phase = q.get('phases', {}).get('search', {})
        if 'durationMs' in s_phase:
            latencies.append(s_phase['durationMs'])
        
        ctxs = s_phase.get('output', {}).get('contexts', [])
        if ctxs:
            context_counts.append(len(ctxs))
            
        # Check evidence hit rate if evidence string / fact is present
        evidence = q.get('evidence', []) or q.get('groundTruth', '')
        if evidence:
            total_with_evidence += 1
            retrieved_text = " ".join([c.get('text', '') if isinstance(c, dict) else str(c) for c in ctxs]).lower()
            if isinstance(evidence, list):
                hit = any(str(e).lower() in retrieved_text for e in evidence)
            else:
                hit = str(evidence).lower() in retrieved_text
            if hit:
                recall_hits += 1

    print("\n==============================================")
    print("      HYBRIDMIND SEARCH METRICS DASHBOARD     ")
    print("==============================================")
    print(f"Total Questions Ingested:        {total_ingested}")
    print(f"Search Phase Completed:          {len(searched)} / {total_ingested} ({len(searched)/total_ingested*100:.1f}%)")
    print("\n--- Breakdown by Conversation ---")
    for c, cnt in sorted(conv_counts.items()):
        print(f"  • {c:12s}: {cnt:3d} questions searched")

    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        p95_lat = sorted(latencies)[int(len(latencies)*0.95)]
        print(f"\n--- Search Latency Statistics ---")
        print(f"  • Average Latency:              {avg_lat:.0f} ms ({avg_lat/1000:.2f}s)")
        print(f"  • P95 Latency:                  {p95_lat:.0f} ms ({p95_lat/1000:.2f}s)")

    if context_counts:
        avg_ctx = sum(context_counts) / len(context_counts)
        print(f"\n--- Context Retrieval Quality ---")
        print(f"  • Average Top-25 Pool Size:     {avg_ctx:.1f} items / query")
        if total_with_evidence > 0:
            print(f"  • Exact Evidence Recall:        {recall_hits}/{total_with_evidence} ({recall_hits/total_with_evidence*100:.1f}%)")

if __name__ == "__main__":
    main()
