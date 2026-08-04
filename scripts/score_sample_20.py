import json
import os
import random
import urllib.request
import urllib.error

ZAI_KEY = os.environ["ZAI_API_KEY"]
ZAI_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

def call_glm46(prompt, max_tokens=500):
    payload = {
        "model": "glm-4.6",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1
    }
    req = urllib.request.Request(
        ZAI_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ZAI_KEY}"
        }
    )
    try:
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            return res["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error calling GLM-4.6: {e}")
        return ""

def main():
    print("Loading checkpoint...")
    with open('/home/azureuser/hybridmind/memorybench/data/runs/hybridmind-locomo-fixed-20260726/checkpoint.json', 'r') as f:
        cp = json.load(f)

    qs = cp.get('questions', {})
    searched = [q for q in qs.values() if q.get('phases', {}).get('search', {}).get('status') == 'completed']

    print(f"Found {len(searched)} searched questions. Selecting 20 random samples...")
    random.seed(42)  # Deterministic seed for reproducible scoring
    sample_20 = random.sample(searched, min(20, len(searched)))

    results = []
    total_score = 0.0

    print("\n" + "="*80)
    print("      SAMPLE EVALUATION (20 RANDOM SEARCHED QUESTIONS)      ")
    print("="*80 + "\n")

    for idx, q in enumerate(sample_20, 1):
        q_id = q.get('id', f'q_{idx}')
        q_text = q.get('question', '') or q.get('query', '')
        ground_truth = q.get('answer', '') or q.get('groundTruth', '') or q.get('evidence', '')

        ctxs = q.get('phases', {}).get('search', {}).get('output', {}).get('contexts', [])
        context_str = "\n".join([f"- {c.get('text', str(c))}" for c in ctxs[:15]])

        # Step 1: Generate Answer using GLM-4.6
        ans_prompt = f"""You are a precise QA assistant answering based strictly on retrieved context memory.
Question: {q_text}

Retrieved Contexts:
{context_str}

Provide a concise, factual answer to the question based on the retrieved context:"""

        generated_answer = call_glm46(ans_prompt, max_tokens=250)

        # Step 2: Judge Answer with GLM-4.6 Judge
        judge_prompt = f"""You are an expert benchmark evaluator.
Question: {q_text}
Ground Truth Answer: {ground_truth}
Candidate Generated Answer: {generated_answer}

Grade whether the Candidate Generated Answer correctly matches the Ground Truth Answer.
Respond ONLY with a JSON object in this exact format:
{{"correct": true, "reason": "brief 1-sentence reason", "score": 1.0}}
(Or {{"correct": false, "reason": "brief 1-sentence reason", "score": 0.0}})"""

        judge_raw = call_glm46(judge_prompt, max_tokens=150)

        score = 0.0
        reason = "Parsing failed"
        try:
            # Parse JSON from response
            cleaned_json = judge_raw.strip()
            if "```json" in cleaned_json:
                cleaned_json = cleaned_json.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_json:
                cleaned_json = cleaned_json.split("```")[1].split("```")[0].strip()

            parsed = json.loads(cleaned_json)
            score = float(parsed.get('score', 1.0 if parsed.get('correct') else 0.0))
            reason = parsed.get('reason', '')
        except Exception:
            if "true" in judge_raw.lower():
                score = 1.0
                reason = "Matched true in response"

        total_score += score
        results.append({
            "id": q_id,
            "question": q_text,
            "ground_truth": str(ground_truth),
            "generated_answer": generated_answer,
            "score": score,
            "reason": reason
        })

        status_symbol = "PASS (1.0)" if score >= 0.5 else "FAIL (0.0)"
        print(f"[{idx}/20] {q_id}: {status_symbol}")
        print(f"   Q: {q_text[:90]}...")
        print(f"   Expected: {str(ground_truth)[:90]}")
        print(f"   Got:      {generated_answer[:90]}")
        print(f"   Reason:   {reason}\n" + "-"*60)

    accuracy = (total_score / len(sample_20)) * 100.0
    print("\n" + "="*80)
    print(f"  FINAL SAMPLE ACCURACY (20 QUESTIONS): {total_score:.1f} / 20 ({accuracy:.1f}%)")
    print("="*80 + "\n")

    # Save to file
    with open('/tmp/sample_20_results.json', 'w') as out_f:
        json.dump({"accuracy": accuracy, "score": total_score, "results": results}, out_f, indent=2)

if __name__ == "__main__":
    main()
