"""
Demo data loader for HybridMind.
Loads ArXiv papers from Hugging Face datasets.
"""

import sys
import os
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.dependencies import get_db_manager


def load_arxiv_from_huggingface(num_papers: int = 200):
    """Load ArXiv papers from Hugging Face datasets."""
    print("Loading ArXiv dataset from Hugging Face...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system(f"{sys.executable} -m pip install datasets --quiet")
        from datasets import load_dataset
    
    # Load ArXiv dataset - CS papers
    dataset = load_dataset(
        "ccdv/arxiv-classification",
        split="train"
    )
    
    # Filter for CS papers and limit
    papers = []
    seen_titles = set()
    
    for item in dataset:
        if len(papers) >= num_papers:
            break
        
        text = item.get("text", "") or item.get("abstract", "")
        title = text[:100].split(".")[0] if text else "Untitled"
        
        # Skip duplicates
        if title in seen_titles or len(text) < 50:
            continue
        seen_titles.add(title)
        
        # Extract category
        label = item.get("label", 0)
        categories = [
            "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", 
            "cs.IR", "cs.RO", "cs.SE", "cs.DS", "cs.CR"
        ]
        category = categories[label % len(categories)]
        
        papers.append({
            "id": f"arxiv-{len(papers):04d}",
            "text": text[:1000],  # Limit text length
            "metadata": {
                "title": title[:200],
                "category": category,
                "source": "arxiv",
                "tags": [category.split(".")[-1], "research", "paper"]
            }
        })
    
    print(f"Loaded {len(papers)} papers from ArXiv dataset")
    return papers


def generate_citation_edges(papers: list, edge_ratio: float = 2.5):
    """Generate citation relationships between papers using simple random strategy."""
    edges = []
    num_edges = int(len(papers) * edge_ratio)
    
    # Group papers by category for more realistic citations
    by_category = {}
    for p in papers:
        cat = p["metadata"].get("category", "other")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(p["id"])
    edge_types = ["supports", "derived_from", "analogous_to", "depends_on", "led_to"]
    for _ in range(num_edges):
        # 70% chance: cite within same category
        if random.random() < 0.7:
            cat = random.choice(list(by_category.keys()))
            if len(by_category[cat]) >= 2:
                source, target = random.sample(by_category[cat], 2)
            else:
                source = random.choice([p["id"] for p in papers])
                target = random.choice([p["id"] for p in papers])
        else:
            # Cross-category citation
            source = random.choice([p["id"] for p in papers])
            target = random.choice([p["id"] for p in papers])
        
        if source != target:
            edge_type = random.choice(edge_types)
            weight = round(random.uniform(0.5, 1.0), 2)
            
            edge_id = f"edge-{source}-{target}-{edge_type}"
            edges.append({
                "id": edge_id,
                "source": source,
                "target": target,
                "type": edge_type,
                "weight": weight
            })
    
    # Remove duplicates
    seen = set()
    unique_edges = []
    for e in edges:
        key = (e["source"], e["target"], e["type"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)
    
    print(f"Generated {len(unique_edges)} citation edges")
    return unique_edges


def generate_semantic_edges(papers: list, embeddings: dict, similarity_threshold: float = 0.6, max_edges_per_node: int = 5):
    """
    Generate edges based on SEMANTIC SIMILARITY between papers.
    
    This creates more meaningful relationships that demonstrate
    hybrid search value - papers are connected because they're
    actually similar in content, not just random.
    
    Args:
        papers: List of paper dicts
        embeddings: Dict mapping paper_id -> embedding vector
        similarity_threshold: Minimum cosine similarity to create edge
        max_edges_per_node: Maximum outgoing edges per paper
        
    Returns:
        List of edge dicts
    """
    import numpy as np
    
    print(f"Generating semantic edges (threshold={similarity_threshold})...")
    
    edges = []
    paper_ids = [p["id"] for p in papers]
    
    # Build embedding matrix
    embedding_list = [embeddings[pid] for pid in paper_ids]
    embedding_matrix = np.vstack(embedding_list)
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    embedding_matrix = embedding_matrix / np.maximum(norms, 1e-10)
    
    # Compute pairwise similarities
    similarities = np.dot(embedding_matrix, embedding_matrix.T)
    
    # Generate edges based on similarity
    for i, source_id in enumerate(paper_ids):
        # Get similarity scores for this paper
        sim_scores = similarities[i]
        
        # Get top similar papers (excluding self)
        top_indices = np.argsort(sim_scores)[::-1]
        
        edge_count = 0
        for j in top_indices:
            if edge_count >= max_edges_per_node:
                break
            if i == j:
                continue
                
            sim = sim_scores[j]
            if sim < similarity_threshold:
                break
            
            target_id = paper_ids[j]
            
            if sim >= 0.85:
                edge_type = "analogous_to"
                weight = 1.0
            elif sim >= 0.75:
                edge_type = "derived_from"
                weight = 0.9
            elif sim >= 0.65:
                edge_type = "depends_on"
                weight = 0.7
            else:
                edge_type = "supports"
                weight = 0.5
            
            edge_id = f"sem-{source_id}-{target_id}"
            edges.append({
                "id": edge_id,
                "source": source_id,
                "target": target_id,
                "type": edge_type,
                "weight": weight,
                "similarity": round(float(sim), 4)
            })
            edge_count += 1
    
    print(f"Generated {len(edges)} semantic edges")
    
    # Report edge type distribution
    type_counts = {}
    for e in edges:
        t = e["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  Edge types: {type_counts}")
    
    return edges


def generate_hybrid_edges(papers: list, embeddings: dict, edge_ratio: float = 2.5, semantic_ratio: float = 0.6):
    """
    Generate edges using HYBRID strategy: semantic similarity + category-based.
    
    This creates the most realistic citation network:
    - 60% edges from semantic similarity (papers cite similar work)
    - 40% edges from category/random (cross-pollination, serendipity)
    
    Args:
        papers: List of paper dicts
        embeddings: Dict mapping paper_id -> embedding vector
        edge_ratio: Total edges per node
        semantic_ratio: Fraction of edges from semantic similarity
        
    Returns:
        List of edge dicts
    """
    num_total = int(len(papers) * edge_ratio)
    num_semantic = int(num_total * semantic_ratio)
    num_random = num_total - num_semantic
    
    print(f"\nGenerating hybrid edges: {num_semantic} semantic + {num_random} category-based")
    
    # Generate semantic edges
    max_per_node = max(1, num_semantic // len(papers))
    semantic_edges = generate_semantic_edges(
        papers, embeddings, 
        similarity_threshold=0.5, 
        max_edges_per_node=max_per_node
    )
    
    # Generate category-based edges
    random_edges = generate_citation_edges(papers, edge_ratio=num_random/len(papers))
    
    # Combine and deduplicate
    all_edges = semantic_edges + random_edges
    seen = set()
    unique_edges = []
    for e in all_edges:
        key = (e["source"], e["target"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)
    
    print(f"Total unique edges: {len(unique_edges)}")
    return unique_edges


def load_demo_data(num_papers: int = 150, clear_existing: bool = False, use_semantic_edges: bool = True):
    """
    Load demo dataset into HybridMind.
    
    Args:
        num_papers: Number of papers to load
        clear_existing: If True, clear existing data without prompt
        use_semantic_edges: If True, create edges based on semantic similarity (RECOMMENDED)
                           This creates more meaningful relationships for hybrid search demo.
    """
    print("=" * 50)
    print("HybridMind Data Loader")
    print("=" * 50)
    
    # Initialize database
    print("\nInitializing HybridMind...")
    db_manager = get_db_manager()
    
    sqlite_store = db_manager.sqlite_store
    vector_index = db_manager.vector_index
    graph_index = db_manager.graph_index
    embedding_engine = db_manager.embedding_engine
    
    # Check existing data
    existing_nodes = sqlite_store.count_nodes()
    if existing_nodes > 0:
        print(f"\nDatabase already contains {existing_nodes} nodes.")
        if not clear_existing:
            response = input("Clear existing data and reload? (y/n): ")
            if response.lower() != 'y':
                print("Keeping existing data.")
                return
        
        with sqlite_store._get_connection() as conn:
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM nodes")
            conn.commit()
        vector_index.clear()
        graph_index.clear()
    
    # Load papers from Hugging Face
    papers = load_arxiv_from_huggingface(num_papers)
    
    print(f"\nLoading {len(papers)} papers into HybridMind...")
    
    # Generate embeddings for all papers (needed for semantic edges)
    embeddings = {}
    print("  Generating embeddings...")
    
    # Load papers and collect embeddings
    for i, paper in enumerate(papers, 1):
        if i % 20 == 0 or i == len(papers):
            print(f"  Progress: {i}/{len(papers)} papers...")
        
        # Generate embedding
        embedding = embedding_engine.embed(paper["text"])
        embeddings[paper["id"]] = embedding
        
        # Store in SQLite
        sqlite_store.create_node(
            node_id=paper["id"],
            text=paper["text"],
            metadata=paper["metadata"],
            embedding=embedding
        )
        
        # Add to vector index
        vector_index.add(paper["id"], embedding)
        
        # Add to graph index
        graph_index.add_node(paper["id"])
    
    # Generate edges - use semantic similarity or random based on flag
    if use_semantic_edges:
        print("\n🔗 Using SEMANTIC SIMILARITY to create edges (recommended for demos)")
        edges = generate_hybrid_edges(papers, embeddings, edge_ratio=2.5, semantic_ratio=0.6)
    else:
        print("\n🔗 Using category-based random edges")
        edges = generate_citation_edges(papers, edge_ratio=2.5)
    
    print(f"\nLoading {len(edges)} relationships...")
    
    # Load edges
    for i, edge in enumerate(edges, 1):
        if i % 50 == 0 or i == len(edges):
            print(f"  Progress: {i}/{len(edges)} edges...")
        
        # Store in SQLite
        sqlite_store.create_edge(
            edge_id=edge["id"],
            source_id=edge["source"],
            target_id=edge["target"],
            edge_type=edge["type"],
            weight=edge["weight"]
        )
        
        # Add to graph index
        graph_index.add_edge(
            source_id=edge["source"],
            target_id=edge["target"],
            edge_type=edge["type"],
            weight=edge["weight"],
            edge_id=edge["id"]
        )
    
    # Save indexes
    print("\nSaving indexes...")
    db_manager.save_indexes()
    
    # Print summary
    stats = db_manager.get_stats()
    print("\n" + "=" * 50)
    print("✅ Data Loaded Successfully!")
    print("=" * 50)
    print(f"  📄 Nodes: {stats['total_nodes']}")
    print(f"  🔗 Edges: {stats['total_edges']}")
    print(f"  📊 Edge Types: {stats['edge_types']}")
    print(f"  🎯 Vector Index: {stats['vector_index_size']} vectors")
    print(f"  🕸️  Graph Nodes: {stats['graph_node_count']}")
    print("\n🔍 Try these searches:")
    print('  - "deep learning neural networks"')
    print('  - "natural language processing transformers"')
    print('  - "computer vision image classification"')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load demo data into HybridMind")
    parser.add_argument("--papers", type=int, default=150, help="Number of papers to load")
    parser.add_argument("--clear", action="store_true", help="Clear existing data without prompt")
    parser.add_argument("--random-edges", action="store_true", 
                       help="Use random category-based edges instead of semantic similarity (NOT recommended)")
    args = parser.parse_args()
    
    load_demo_data(
        num_papers=args.papers, 
        clear_existing=args.clear,
        use_semantic_edges=not args.random_edges
    )
