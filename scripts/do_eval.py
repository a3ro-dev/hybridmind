import json, random, urllib.request, time, os

data = json.load(open('/home/azureuser/hybridmind/memorybench/data/benchmarks/locomo/locomo10.json'))

all_qa = []
for elem in data[:5]:
    sid = elem.get('sample_id')
    for q_item in elem.get('qa', []):
        all_qa.append((sid, q_item.get('question'), str(q_item.get('answer'))))

random.seed(42)
sample_20 = random.sample(all_qa, min(20, len(all_qa)))

print(f"=== Starting Manual Search + Answer + LLM Judge Evaluation on {len(sample_20)} Random Questions ===")

API_KEY = os.environ["ZAI_API_KEY"]
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

def call_llm(prompt, sys_prompt="You are a helpful assistant."):
    payload = json.dumps({
        "model": "glm-4.6",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }).encode('utf-8')
    req = urllib.request.Request(BASE_URL, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + API_KEY
    })
    try:
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            return res['choices'][0]['message']['content'].strip()
    except Exception as e:
        return "Error: " + str(e)

correct_count = 0
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
            for r in s_data.get('results', [])[:25]:
                ctx_snippets.append(r.get('text', ''))
    except Exception as e:
        ctx_snippets = ["Search error: " + str(e)]

    context_str = "\n".join(["- " + snippet[:250] for snippet in ctx_snippets[:15]])

    ans_prompt = "Context:\n" + context_str + "\n\nQuestion: " + q_text + "\n\nProvide a concise, direct answer based strictly on the context above."
    generated_ans = call_llm(ans_prompt)

    judge_prompt = "Question: " + q_text + "\nGround Truth Answer: " + gt_ans + "\nGenerated Model Answer: " + generated_ans + "\n\nDoes the Generated Model Answer correctly state or entail the Ground Truth Answer?\nRespond ONLY with either CORRECT or INCORRECT."
    judge_res = call_llm(judge_prompt, "You are an impartial judge evaluating question answering accuracy.")

    is_correct = "CORRECT" in judge_res.upper()
    if is_correct:
        correct_count += 1

    print(f"[{i:02d}/20] ({sid}) Question: {q_text}")
    print(f"       Ground Truth  : {gt_ans}")
    print(f"       Generated Ans : {generated_ans[:100]}")
    print(f"       Judge Grade   : {'CORRECT' if is_correct else 'INCORRECT'}")
    print("-" * 70)
    time.sleep(0.5)

print("=" * 70)
print(f"FINAL ACCURACY SCORE: {correct_count} / {len(sample_20)} ({correct_count/len(sample_20)*100:.1f}%)")
print("=" * 70)
