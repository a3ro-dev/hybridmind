import json

path = '/home/azureuser/hybridmind/memorybench/data/runs/hybridmind-locomo-fixed-20260726/checkpoint.json'
cp = json.load(open(path))
qs = cp.get('questions', {})
s_done = sum(1 for q in qs.values() if q.get('phases', {}).get('search', {}).get('status') == 'completed')
a_done = sum(1 for q in qs.values() if q.get('phases', {}).get('answer', {}).get('status') == 'completed')
e_done = sum(1 for q in qs.values() if q.get('phases', {}).get('evaluate', {}).get('status') == 'completed')
print(f"Total: {len(qs)} | Search Done: {s_done} | Answer Done: {a_done} | Eval Done: {e_done}")
