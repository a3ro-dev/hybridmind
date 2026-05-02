#!/bin/bash
set -e

echo "Running Retrieval Ablation..."
python scripts/retrieval_ablation.py

echo "Running Targeted Graph Benchmark..."
python scripts/targeted_graph_benchmark.py

echo "Running Ingest Ablation..."
python scripts/ingest_ablation.py

echo "All benchmarks completed. Results saved to benchmarks/results/"
