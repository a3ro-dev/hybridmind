import json
import logging
import time
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# To test ingest neighborhood averaging, we ingest a core set, then a probe.
# We test vector search on the probe with and without neighborhood averaging.

CORE_DATASET = [
    {"id": "c1", "text": "A comprehensive study on climate change and ocean temperatures.", "metadata": {"domain": "science"}},
    {"id": "c2", "text": "Rising sea levels threaten coastal communities globally.", "metadata": {"domain": "science"}},
    {"id": "c3", "text": "The impact of carbon emissions on global warming.", "metadata": {"domain": "science"}},
    {"id": "c4", "text": "Melting glaciers and their effect on oceanic currents.", "metadata": {"domain": "science"}},
]

PROBE = {"id": "p1", "text": "Environmental policies for mitigating atmospheric temperature increases.", "metadata": {"domain": "policy"}}
# The probe is about policy, but semantically related to science topics.
# With neighborhood averaging, its embedding should shift closer to the science core.

QUERIES = [
    # Semantic queries to retrieve the probe
    "environmental policy",
    "global warming mitigation",
    "climate change"
]

def run_ingest_ablation():
    client = TestClient(app)

    results = {}

    for averaging_enabled in [False, True]:
        settings.use_graph_conditioned_embeddings = averaging_enabled
        client.delete("/bulk/clear")
        time.sleep(1)

        for doc in CORE_DATASET:
            client.post("/nodes", json={"text": doc["text"], "metadata": doc["metadata"]})

        # Ingest probe
        res = client.post("/nodes", json={"text": PROBE["text"], "metadata": PROBE["metadata"]})
        probe_id = res.json()["id"]

        # We retrieve the probe itself to see its cosine diff with pure raw embedding
        probe_node = client.get(f"/nodes/{probe_id}").json()

        # Test how often probe is retrieved in top 2
        retrieved_count = 0
        for q in QUERIES:
            res = client.post("/search/vector", json={"query_text": q, "top_k": 2})
            r_ids = [r["node_id"].split("_")[0] for r in res.json()["results"]]
            if probe_id in r_ids:
                retrieved_count += 1

        results["averaging_enabled" if averaging_enabled else "averaging_disabled"] = {
            "probe_retrieved_top2_count": retrieved_count,
            "total_queries": len(QUERIES)
        }

    output_path = Path("benchmarks/results/ingest_ablation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Ingest ablation results saved to {output_path}")
    logger.info(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_ingest_ablation()
