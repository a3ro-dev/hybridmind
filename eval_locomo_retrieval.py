"""
Quick retrieval-only LoCoMo evaluation.
Calls the live HybridMind /search/hybrid endpoint for a sample of questions
and computes Hit@k / MRR using answer-text overlap as relevance signal.
"""
import json
import sys
import time
import re
import httpx
from pathlib import Path
from statistics import mean

BASE_URL = "http://127.0.0.1:8000"
LOCOMO_PATH = Path("memorybench/data/benchmarks/locomo/locomo10.json")
SAMPLES_PER_CATEGORY = 5

CATEGORY_MAP = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "world-knowledge", 5: "adversarial"}

def load_questions():
    """Load up to 5 questions per category from LoCoMo dataset."""
    data = json.loads(LOCOMO_PATH.read_text())
    questions = []
    counts = {}
    for conv in data:
        for qa in conv.get("qa", []):
            cat = CATEGORY_MAP.get(qa.get("category", 0), "unknown")
            if counts.get(cat, 0) >= SAMPLES_PER_CATEGORY:
                continue
            counts[cat] = counts.get(cat, 0) + 1
            questions.append({
                "question": qa["question"],
                "answer": str(qa["answer"]).strip(),
                "category": cat,
                "evidence": qa.get("evidence", []),
            })
    return questions

def answer_tokens(answer: str) -> set:
    """Tokenize answer into meaningful search tokens."""
    tokens = set(re.findall(r"[A-Za-z0-9']+", answer.lower()))
    # Remove very common stop words
    stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "is", "was", "it", "and", "or", "but"}
    return tokens - stopwords

def is_relevant(retrieved_text: str, answer: str) -> bool:
    """Check if retrieved text contains the answer (or key tokens).
    
    Three-tier check:
    1. Exact answer substring match (e.g., "7 May 2023" in text)
    2. All significant answer tokens appear in text
    3. Partial token overlap for longer answers
    """
    text_lower = retrieved_text.lower()
    answer_lower = answer.lower()
    
    # Tier 1: Direct substring match
    if answer_lower in text_lower:
        return True
    
    # Tier 2: All significant tokens present
    answer_toks = answer_tokens(answer)
    if not answer_toks:
        return False
    text_toks = set(re.findall(r"[A-Za-z0-9']+", text_lower))
    
    # For short answers (1-3 tokens), require all tokens
    if len(answer_toks) <= 3:
        return answer_toks.issubset(text_toks)
    
    # For longer answers, require 70%+ token overlap
    overlap = len(answer_toks & text_toks)
    return overlap / len(answer_toks) >= 0.7

def evaluate():
    questions = load_questions()
    
    if not questions:
        print("ERROR: No questions loaded from LoCoMo data")
        sys.exit(1)
    
    print(f"Loaded {len(questions)} questions for evaluation:")
    cats = {}
    for q in questions:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    print()
    
    client = httpx.Client(base_url=BASE_URL, timeout=30.0)
    
    all_hits_1 = []
    all_hits_5 = []
    all_hits_10 = []
    all_mrr = []
    all_prec_5 = []
    all_prec_10 = []
    
    results_by_category = {}
    
    for i, q in enumerate(questions):
        question = q["question"]
        answer = q["answer"]
        category = q["category"]
        
        print(f"[{i+1}/{len(questions)}] [{category}] {question[:80]}...")
        print(f"  Answer: {answer[:80]}...")
        
        try:
            resp = client.post("/search/hybrid", json={
                "query_text": question,
                "top_k": 15,
                "min_score": 0.0,
            })
            if resp.status_code != 200:
                print(f"  ERROR: HTTP {resp.status_code}")
                continue
                
            data = resp.json()
            results = data.get("results", [])
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        
        # Evaluate at k=1,5,10
        for k, hits_list, prec_list in [
            (1, all_hits_1, None),
            (5, all_hits_5, all_prec_5),
            (10, all_hits_10, all_prec_10),
        ]:
            top_k = results[:k]
            relevant = [r for r in top_k if is_relevant(r.get("text", ""), answer)]
            hit = 1.0 if relevant else 0.0
            hits_list.append(hit)
            
            if prec_list is not None:
                prec_list.append(len(relevant) / k)
        
        # MRR
        mrr = 0.0
        for rank, r in enumerate(results[:10], 1):
            if is_relevant(r.get("text", ""), answer):
                mrr = 1.0 / rank
                break
        all_mrr.append(mrr)
        
        print(f"  Hit@1={all_hits_1[-1]:.0%} Hit@5={all_hits_5[-1]:.0%} Hit@10={all_hits_10[-1]:.0%} MRR={mrr:.3f}")
        
        # Show top 3 results for debugging
        for rank, r in enumerate(results[:3], 1):
            text = r.get("text", "")[:120]
            vs = r.get("vector_score", 0)
            gs = r.get("graph_score", 0)
            gg = r.get("graph_gate", 1.0)
            egs = r.get("effective_graph_score", gs)
            cs = r.get("combined_score", 0)
            rel = "✓" if is_relevant(text, answer) else "✗"
            print(f"    #{rank} {rel} v={vs:.3f} g={gs:.3f} gate={gg:.3f} efg={egs:.3f} c={cs:.3f} {text[:100]}")
        
        # Per-category tracking
        if category not in results_by_category:
            results_by_category[category] = {"hits": [], "mrrs": []}
        results_by_category[category]["hits"].append(all_hits_5[-1])
        results_by_category[category]["mrrs"].append(mrr)
        
        time.sleep(0.1)  # Rate limiting
    
    client.close()
    
    # Compute aggregate metrics
    n = len(all_hits_1)
    print("\n" + "=" * 70)
    print("  RETRIEVAL EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Questions evaluated: {n}")
    print(f"  Hit@1:  {mean(all_hits_1):.4f}  ({sum(all_hits_1)}/{n})")
    print(f"  Hit@5:  {mean(all_hits_5):.4f}  ({sum(all_hits_5)}/{n})")
    print(f"  Hit@10: {mean(all_hits_10):.4f}  ({sum(all_hits_10)}/{n})")
    print(f"  Precision@5:  {mean(all_prec_5):.4f}")
    print(f"  Precision@10: {mean(all_prec_10):.4f}")
    print(f"  MRR:    {mean(all_mrr):.4f}")
    print()
    
    print("  --- By Category ---")
    for cat in sorted(results_by_category.keys()):
        c = results_by_category[cat]
        print(f"  {cat}: Hit@5={mean(c['hits']):.4f} MRR={mean(c['mrrs']):.4f} (n={len(c['hits'])})")
    
    print()
    # Quick summary line for iteration tracking
    print(f"  SUMMARY: Hit@1={mean(all_hits_1):.3f} Hit@5={mean(all_hits_5):.3f} Hit@10={mean(all_hits_10):.3f} Prec@5={mean(all_prec_5):.3f} Prec@10={mean(all_prec_10):.3f} MRR={mean(all_mrr):.3f}")
    
    return {
        "hit_at_1": mean(all_hits_1),
        "hit_at_5": mean(all_hits_5),
        "hit_at_10": mean(all_hits_10),
        "precision_at_5": mean(all_prec_5),
        "precision_at_10": mean(all_prec_10),
        "mrr": mean(all_mrr),
        "n": n,
    }

if __name__ == "__main__":
    evaluate()
