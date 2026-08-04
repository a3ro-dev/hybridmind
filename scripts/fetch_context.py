import json, random, urllib.request

data = json.load(open('/home/azureuser/hybridmind/memorybench/data/benchmarks/locomo/locomo10.json'))

all_qa = []
for elem in data[:5]:
    sid = elem.get('sample_id')
    for q_item in elem.get('qa', []):
        all_qa.append((sid, q_item.get('question'), str(q_item.get('answer'))))

random.seed(42)
sample_20 = random.sample(all_qa, min(20, len(all_qa)))

results = []
for i, (sid, q_text, gt_ans) in enumerate(sample_20, 1):
    s_req = urllib.request.Request(
        'http://127.0.0.1:8010/search/hybrid',
        data=json.dumps({'query_text': q_text, 'top_k': 25}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    ctx_snippets = []
    try:
        with urllib.request.urlopen(s_req) as s_resp:
            s_data = json.loads(s_resp.read().decode('utf-8'))
            for r in s_data.get('results', []):
                ctx_snippets.append(r.get('text', ''))
    except Exception as e:
        ctx_snippets = [f"Search error: {e}"]
        
    results.append({
        "index": i,
        "sample_id": sid,
        "question": q_text,
        "ground_truth": gt_ans,
        "top_25_context": ctx_snippets
    })

with open('/tmp/20_sample_context.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Successfully retrieved Top-25 search context for {len(results)} questions!")
