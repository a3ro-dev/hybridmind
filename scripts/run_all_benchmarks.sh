#!/bin/bash
set -e

echo "Running Retrieval Ablation..."
python scripts/retrieval_ablation.py

echo "Running Targeted Graph Benchmark..."
python scripts/targeted_graph_benchmark.py

echo "Running Ingest Ablation..."
python scripts/ingest_ablation.py

echo "Running LoCoMo Retrieval Eval..."
python eval_locomo_retrieval.py --n 10 || echo "LoCoMo eval failed (server may be down)"

echo "Running LongMemEval Retrieval Eval..."
python eval_longmemeval_retrieval.py --n 20 || echo "LongMemEval eval skipped (data may not be downloaded)"

echo "Running MuSiQue Multi-Hop Retrieval Eval..."
python eval_musique_retrieval.py --n 50 || echo "MuSiQue eval skipped (data may not be downloaded)"

echo ""
echo "All benchmarks completed. Results saved to benchmarks/results/"
echo ""
echo "To download missing benchmark data:"
echo "  LongMemEval: https://github.com/tiger-ai-lab/LongMemEval"
echo "  MuSiQue:     python benchmarks/data/musique/download_musique.py"
