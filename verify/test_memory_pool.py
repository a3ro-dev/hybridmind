"""
Verify memory pool classification rules.
"""
from models.memory_pool import classify_memory_type, MemoryPool

def test_pool_classification():
    # 1. SUMMARY
    assert classify_memory_type("Generally speaking, Alice prefers working in the mornings.") == MemoryPool.SUMMARY
    assert classify_memory_type("Overall Bob has attended 5 meetings this week.") == MemoryPool.SUMMARY

    # 2. EVENTS
    assert classify_memory_type("Alice visited the Seattle office on July 2nd, 2026.") == MemoryPool.EVENTS
    assert classify_memory_type("Bob is scheduled to fly next Monday.") == MemoryPool.EVENTS

    # 3. NOTES
    assert classify_memory_type("I think this new framework is highly efficient.") == MemoryPool.NOTES
    assert classify_memory_type("Bob prefers dark mode for IDEs.") == MemoryPool.NOTES

    # 4. RAW
    assert classify_memory_type("This is a simple fact statement.") == MemoryPool.RAW
