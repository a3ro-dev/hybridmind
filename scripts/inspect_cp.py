import json

with open('/home/azureuser/hybridmind/memorybench/data/runs/hybridmind-locomo-fixed-20260726/checkpoint.json', 'r') as f:
    cp = json.load(f)

for k, q in cp['questions'].items():
    s = q.get('phases', {}).get('search', {})
    if s.get('status') == 'completed':
        print("resultFile:", s.get('resultFile'))
        print("results type:", type(s.get('results')))
        print("results sample:", str(s.get('results'))[:300])
        break
