"""Debug: trace graph expansion for the edge-dependent multi-hop benchmark case."""
import json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from main import app

DATASET = [
    {"id": "sem1", "text": "Canines are known to be loyal companions.", "metadata": {"domain": "misc"}},
    {"id": "sem2", "text": "Felines often enjoy sleeping in sunny spots.", "metadata": {"domain": "misc"}},
    {"id": "lex1", "text": "The patient was prescribed amoxapine 50mg for depression.", "metadata": {"domain": "medical"}},
    {"id": "lex2", "text": "The patient was prescribed fluoxetine 20mg for depression.", "metadata": {"domain": "medical"}},
    {"id": "nodeA", "text": "Company X announced a breakthrough in quantum computing.", "metadata": {"domain": "news"}},
    {"id": "nodeB", "text": "Dr. Smith leads the team.", "metadata": {"domain": "news"}},
    {"id": "nodeC", "text": "The new processor operates at near absolute zero.", "metadata": {"domain": "news"}},
    {"id": "pad1", "text": "Who is the leader of the engineering team?", "metadata": {"domain": "misc"}},
    {"id": "pad2", "text": "Processors are becoming faster every year.", "metadata": {"domain": "misc"}},
]

EDGES = [
    ("nodeA", "nodeB", "supports", 1.0),
    ("nodeB", "nodeC", "supports", 1.0),
]

client = TestClient(app)
client.delete("/bulk/clear")
time.sleep(1)

dataset_ids_map = {}
for doc in DATASET:
    res = client.post("/nodes", json={"text": doc["text"], "metadata": doc["metadata"]})
    node_id = res.json()["id"]
    dataset_ids_map[doc["id"]] = node_id

for source, target, edge_type, weight in EDGES:
    source_id = dataset_ids_map[source]
    target_id = dataset_ids_map[target]
    r = client.post("/edges", json={"source_id": source_id, "target_id": target_id, "type": edge_type, "weight": weight})
    print(f"Edge: {source} -> {target} => {r.status_code}")

from api.dependencies import get_db_manager
db = get_db_manager()
nodeA_id = dataset_ids_map["nodeA"]
nodeB_id = dataset_ids_map["nodeB"]
nodeC_id = dataset_ids_map["nodeC"]
print(f"\nGraph has_node(nodeA): {db.graph_index.has_node(nodeA_id)}")
print(f"Graph has_node(nodeB): {db.graph_index.has_node(nodeB_id)}")
print(f"Graph has_node(nodeC): {db.graph_index.has_node(nodeC_id)}")

from storage.graph_index import GraphIndex
gi = db.graph_index
print(f"Graph edges: {gi.edge_count}")
for u, v, data in gi.graph.edges(data=True):
    print(f"  {u[-6:]} -> {v[-6:]}  type={data.get('type')} weight={data.get('weight')}")

neighbors, _, _ = db.graph_engine.traverse(start_id=nodeA_id, depth=2)
print(f"\nTraverse from nodeA (depth=2): found {len(neighbors)} nodes")
for n in neighbors:
    print(f"  {n['node_id'][-6:]} dist={n['depth']} path={[p[-6:] for p in n['path']]}")

refs = [nodeA_id]
all_ids = [nodeA_id, nodeB_id, nodeC_id, dataset_ids_map["pad1"], dataset_ids_map["pad2"]]
prox = db.graph_engine.compute_proximity_scores(all_ids, refs, max_depth=2)
print(f"\nProximity scores (ref=nodeA):")
for nid, score in prox.items():
    label = next((k for k, v in dataset_ids_map.items() if v == nid), "?")
    print(f"  {label}: {score}")

query = "What leads the engineering team"
payload = {
    "query_text": query,
    "top_k": 3,
    "vector_weight": 0.1,
    "graph_weight": 0.9,
    "max_depth": 2,
    "anchor_nodes": [nodeA_id],
}
res = client.post("/search/hybrid", json=payload)
results = res.json().get("results", [])
print(f"\nSearch: '{query}' (vw=0.1, gw=0.9, anchor=nodeA)")
for i, r in enumerate(results[:5]):
    nid = r["node_id"]
    label = next((k for k, v in dataset_ids_map.items() if v == nid), "?")
    print(f" #{i+1} c={r['combined_score']:.4f} v={r['vector_score']:.4f} g={r['graph_score']:.4f}  [{label}] {r['text'][:80]}")

s = client.get("/search/stats").json()
print(f"\nStats: nodes={s['total_nodes']} edges={s['total_edges']} graph_nodes={s['graph_node_count']} graph_edges={s['graph_edge_count']}")
