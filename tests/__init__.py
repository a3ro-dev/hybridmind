"""
HybridMind Test Suite

Based on Test Case Markers for:
"Vector + Graph Native Database for Efficient AI Retrieval"

Test Files:
-----------
- test_api_crud.py      : TC-API-01 to TC-API-05 (Node/Edge CRUD)
- test_vector_search.py : TC-VEC-01 to TC-VEC-03 (Vector search)
- test_graph_traversal.py: TC-GRAPH-01 to TC-GRAPH-03 (Graph traversal)
- test_hybrid_search.py : TC-HYB-01 to TC-HYB-03 (Hybrid search)
- test_canonical_dataset.py: Section 10 (Canonical dataset tests)

Usage:
------
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api_crud.py -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html
"""
