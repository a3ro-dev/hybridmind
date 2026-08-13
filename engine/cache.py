"""
Query result caching for HybridMind.
Implements LRU cache with TTL for fast repeated queries.
"""

import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class QueryCache:
    """
    Thread-safe query result cache with TTL support.

    Features:
    - LRU eviction when max size reached
    - Time-based expiration (TTL)
    - Hit/miss statistics
    - Cache invalidation on mutations
    """

    def __init__(self, maxsize: int = 1000, ttl: int = 300):
        """
        Initialize query cache.

        Args:
            maxsize: Maximum number of cached entries
            ttl: Time-to-live in seconds (default 5 minutes)
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}  # For LRU tracking
        self.lock = threading.Lock()

        # Statistics
        self.hits = 0
        self.misses = 0

        logger.info(f"QueryCache initialized: maxsize={maxsize}, ttl={ttl}s")

    def _make_key(self, query_type: str, params: dict) -> str:
        """
        Create deterministic cache key from query type and parameters.

        Args:
            query_type: Type of query (vector, graph, hybrid)
            params: Query parameters

        Returns:
            MD5 hash of the query signature
        """
        # Sort params for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True, default=str)
        key_str = f"{query_type}:{sorted_params}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query_type: str, params: dict) -> Optional[dict]:
        """
        Get cached result if available and not expired.

        Args:
            query_type: Type of query
            params: Query parameters

        Returns:
            Cached result or None if not found/expired
        """
        key = self._make_key(query_type, params)

        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            entry = self.cache[key]

            # Check TTL
            if time.time() - entry["timestamp"] > self.ttl:
                # Expired - remove and return None
                del self.cache[key]
                del self.access_times[key]
                self.misses += 1
                return None

            # Update access time for LRU
            self.access_times[key] = time.time()
            self.hits += 1

            return entry["result"]

    def set(self, query_type: str, params: dict, result: dict):
        """
        Cache a query result.

        Args:
            query_type: Type of query
            params: Query parameters
            result: Result to cache
        """
        key = self._make_key(query_type, params)
        now = time.time()

        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.maxsize and key not in self.cache:
                self._evict_lru()

            self.cache[key] = {"result": result, "timestamp": now}
            self.access_times[key] = now

    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.access_times:
            return

        # Find LRU key
        lru_key = min(self.access_times, key=self.access_times.get)

        del self.cache[lru_key]
        del self.access_times[lru_key]

    def invalidate_all(self):
        """
        Clear all cached entries.
        Call this after node/edge mutations.
        """
        with self.lock:
            cleared = len(self.cache)
            self.cache.clear()
            self.access_times.clear()
            logger.info(f"Cache invalidated: {cleared} entries cleared")

    def invalidate_pattern(self, query_type: str):
        """
        Invalidate entries matching a query type pattern.

        Args:
            query_type: Type of queries to invalidate
        """
        # Keys intentionally contain only a digest, so selective invalidation is
        # not possible without changing the on-disk/API contract.  Clear the
        # cache directly while holding the lock.  Calling invalidate_all() here
        # would try to acquire the same non-reentrant lock twice and deadlock.
        with self.lock:
            cleared = len(self.cache)
            self.cache.clear()
            self.access_times.clear()
            logger.info(
                "Cache pattern invalidation (%s): %d entries cleared",
                query_type,
                cleared,
            )

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        # Calculate average TTL remaining
        now = time.time()
        ttl_remaining = []
        with self.lock:
            for entry in self.cache.values():
                remaining = self.ttl - (now - entry["timestamp"])
                if remaining > 0:
                    ttl_remaining.append(remaining)

        avg_ttl = sum(ttl_remaining) / len(ttl_remaining) if ttl_remaining else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 4),
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "ttl_seconds": self.ttl,
            "avg_ttl_remaining": round(avg_ttl, 1),
        }

    def cleanup_expired(self):
        """Remove all expired entries."""
        now = time.time()
        expired = []

        with self.lock:
            for key, entry in self.cache.items():
                if now - entry["timestamp"] > self.ttl:
                    expired.append(key)

            for key in expired:
                del self.cache[key]
                del self.access_times[key]

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired cache entries")

        return len(expired)


# Singleton instance for global use
_query_cache: Optional[QueryCache] = None


def get_query_cache(maxsize: int = 1000, ttl: int = 300) -> QueryCache:
    """
    Get or create the query cache singleton.

    Args:
        maxsize: Maximum cache size
        ttl: Time-to-live in seconds

    Returns:
        QueryCache instance
    """
    global _query_cache

    if _query_cache is None:
        _query_cache = QueryCache(maxsize=maxsize, ttl=ttl)

    return _query_cache


def invalidate_cache():
    """Invalidate the global cache (call after mutations)."""
    if _query_cache is not None:
        _query_cache.invalidate_all()
