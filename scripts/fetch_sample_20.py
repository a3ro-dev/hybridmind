import json
import random

def main():
    with open('/home/azureuser/hybridmind/memorybench/data/runs/hybridmind-locomo-fixed-20260726/checkpoint.json', 'r') as f:
        cp = json.load(f)

    qs = cp.get('questions', {})
    searched = [q for q in qs.values() if q.get('phases', {}).get('search', {}).get('status') == 'completed']
    
    random.seed(42)
    sample_20 = random.sample(searched, min(20, len(searched)))

    data = []
    for q in sample_20:
        s_phase = q.get('phases', {}).get('search', {})
        results_list = s_phase.get('results', [])
        
        ctx_texts = []
        for item in results_list[:15]:
            if isinstance(item, dict):
                ctx_texts.append(item.get('text', '') or str(item))
            else:
                ctx_texts.append(str(item))
                
        data.append({
            "id": q.get('id', ''),
            "question": q.get('question', '') or q.get('query', ''),
            "ground_truth": q.get('answer', '') or q.get('groundTruth', '') or q.get('evidence', ''),
            "contexts": ctx_texts
        })

    with open('/tmp/sample_20_raw.json', 'w') as out_f:
        json.dump(data, out_f, indent=2)
    print(f"Exported {len(data)} sample questions with {sum(len(d['contexts']) for d in data)} total contexts to /tmp/sample_20_raw.json")

if __name__ == "__main__":
    main()
