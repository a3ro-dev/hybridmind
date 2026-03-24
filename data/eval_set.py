import sqlite3
import json
import random
import re
from pathlib import Path

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def main():
    db_path = Path("data/hybridmind.mind/store.db")
    if not db_path.exists():
        print(f"Database {db_path} not found. Load data first.")
        return
        
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    # Handle both new soft-delete schema and old schema in case it hasn't migrated yet
    try:
        cursor.execute("SELECT id, text FROM nodes WHERE deleted_at IS NULL")
    except sqlite3.OperationalError:
        cursor.execute("SELECT id, text FROM nodes")
        
    nodes = cursor.fetchall()
    conn.close()
    
    if not nodes:
        print("No nodes found in database.")
        return
        
    print(f"Loaded {len(nodes)} nodes for evaluation set generation.")
    
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("Please install rank_bm25: pip install rank_bm25")
        return
        
    documents = [tokenize(row[1]) for row in nodes]
    node_ids = [row[0] for row in nodes]
    
    print("Fitting BM25 baseline...")
    bm25 = BM25Okapi(documents)
    
    queries = []
    random.seed(42)  # For reproducibility
    
    print("Generating 50 queries...")
    for _ in range(50):
        source_idx = random.randint(0, len(nodes) - 1)
        source_text = nodes[source_idx][1]
        
        # Extract a random query-like chunk (3-6 words)
        words = source_text.split()
        if len(words) < 6:
            query = source_text
        else:
            start = random.randint(0, len(words) - 6)
            query = " ".join(words[start:start+random.randint(3, 6)])
            
        # Get ground truth from BM25
        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        
        # Top 5 are ground truth
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        ground_truth_ids = [node_ids[i] for i in top_k_indices if scores[i] > 0]
        
        queries.append({
            "query": query,
            "ground_truth_ids": ground_truth_ids
        })
        
    out_path = Path("data/eval_queries.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"queries": queries}, f, indent=2)
        
    print(f"Generated 50 queries in {out_path}")

if __name__ == "__main__":
    main()
