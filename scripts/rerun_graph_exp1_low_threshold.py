"""
Delete all edges, rebuild cross-domain graph at a lower cosine threshold, re-run Experiment 1.
Does not reload datasets. Appends results to benchmarks/multi_domain_results.json under threshold_0_20_rerun.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.multi_domain_eval import BASE_URL, MultiDomainEval, RESULTS_FILE  # noqa: E402

THRESHOLD = 0.20


def main() -> None:
    prev = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    e = MultiDomainEval(skip_lock=True)
    e.edge_threshold = THRESHOLD
    e.results["metadata"]["edge_threshold"] = THRESHOLD
    e.results["metadata"]["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    e.results["loading"] = prev.get("loading", {})
    stats_before = e._json(e.client.get(f"{BASE_URL}/search/stats"))
    n_edges_before = int(stats_before.get("total_edges", 0))
    deleted = e.delete_all_edges()
    e.fetch_all_nodes()
    e.build_cross_domain_graph()
    e.run_experiment_1()
    gc = e.results["graph_construction"]
    exp1 = e.results["experiments"]["exp1"]
    total_edges = gc["total_cross_domain_edges"]
    diff_any = any(r["different_results_count"] > 0 for r in exp1["queries"])
    diff_domains = sum(
        1 for r in exp1["queries"] if r["vector_domain_distribution"] != r["hybrid_domain_distribution"]
    )
    rerun = {
        "edge_threshold": THRESHOLD,
        "edges_deleted_before_rebuild": deleted,
        "edges_in_index_before_delete": n_edges_before,
        "graph_construction": gc,
        "experiments": {"exp1": exp1},
        "summary": {
            "total_cross_domain_edges": total_edges,
            "exp1_any_top10_set_difference": diff_any,
            "exp1_queries_with_domain_distribution_change": diff_domains,
        },
    }
    prev["threshold_0_20_rerun"] = rerun
    RESULTS_FILE.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    print(json.dumps(rerun["summary"], indent=2))
    print(f"total_cross_domain_edges={total_edges}")
    print(f"Wrote threshold_0_20_rerun to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
