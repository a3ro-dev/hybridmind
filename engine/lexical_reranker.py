"""Query-local lexical reranking for a bounded hybrid candidate pool."""
from __future__ import annotations

import math
from collections import Counter, OrderedDict
from threading import RLock
from typing import Any, Callable, Dict, List, Optional


class BoundedTokenSetCache:
    """Thread-safe LRU cache for tokenizer output used by lexical reranking."""

    def __init__(self, tokenize: Callable[[str], List[str]], max_entries: int = 20_000):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._tokenize = tokenize
        self._max_entries = max_entries
        self._values: OrderedDict[str, frozenset[str]] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0

    def get(self, text: str) -> frozenset[str]:
        with self._lock:
            cached = self._values.get(text)
            if cached is not None:
                self._values.move_to_end(text)
                self._hits += 1
                return cached

        token_set = frozenset(self._tokenize(text))
        with self._lock:
            # Another thread may have populated the same text while tokenizing.
            cached = self._values.get(text)
            if cached is not None:
                self._values.move_to_end(text)
                self._hits += 1
                return cached
            self._values[text] = token_set
            self._values.move_to_end(text)
            self._misses += 1
            while len(self._values) > self._max_entries:
                self._values.popitem(last=False)
            return token_set

    def info(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._values),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
            }


def rerank_with_query_local_lexical_rrf(
    query_text: str,
    candidates: List[Dict[str, Any]],
    tokenize: Callable[[str], List[str]],
    *,
    pool_size: int = 500,
    lexical_weight: float = 0.5,
    rrf_k: int = 60,
    document_term_cache: Optional[BoundedTokenSetCache] = None,
) -> List[Dict[str, Any]]:
    """Fuse the existing candidate rank with a query-local lexical rank.

    The lexical score uses query-term inverse document frequency within the
    candidate pool and square-root document-length normalization. Only the
    bounded pool is returned because downstream rerankers consume its top-N.
    """
    if not candidates or pool_size <= 0:
        return []
    if not 0.0 <= lexical_weight <= 1.0:
        raise ValueError("lexical_weight must be between 0 and 1")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")

    base_ranked = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.get("combined_score", 0.0)),
            -float(candidate.get("vector_score", 0.0)),
        ),
    )[:pool_size]
    query_terms = tuple(dict.fromkeys(tokenize(query_text)))
    if not query_terms or len(base_ranked) == 1:
        return base_ranked

    if document_term_cache is not None:
        document_terms = [
            document_term_cache.get(str(candidate.get("text", "")))
            for candidate in base_ranked
        ]
    else:
        document_terms = [set(tokenize(str(candidate.get("text", "")))) for candidate in base_ranked]
    document_frequency = Counter(
        term
        for terms in document_terms
        for term in query_terms
        if term in terms
    )
    candidate_count = len(base_ranked)
    lexical_scores = []
    for terms in document_terms:
        score = sum(
            math.log((candidate_count + 1) / (document_frequency[term] + 1)) + 1.0
            for term in query_terms
            if term in terms
        )
        lexical_scores.append(score / math.sqrt(max(len(terms), 1)))

    lexical_order = sorted(
        range(candidate_count),
        key=lambda index: (
            -lexical_scores[index],
            -float(base_ranked[index].get("combined_score", 0.0)),
        ),
    )
    lexical_ranks = {candidate_index: rank for rank, candidate_index in enumerate(lexical_order, 1)}
    base_weight = 1.0 - lexical_weight

    for candidate_index, candidate in enumerate(base_ranked):
        base_rank = candidate_index + 1
        lexical_rank = lexical_ranks[candidate_index]
        fused_score = (
            base_weight / (rrf_k + base_rank)
            + lexical_weight / (rrf_k + lexical_rank)
        )
        candidate["pre_lexical_combined_score"] = float(candidate.get("combined_score", 0.0))
        candidate["lexical_rerank_score"] = lexical_scores[candidate_index]
        candidate["combined_score"] = fused_score
        candidate["reasoning"] = (
            f"{candidate.get('reasoning', '')}; "
            f"local_lexical_rrf(base_rank={base_rank}, lexical_rank={lexical_rank})"
        ).lstrip("; ")

    return sorted(
        base_ranked,
        key=lambda candidate: (
            -float(candidate.get("combined_score", 0.0)),
            -float(candidate.get("pre_lexical_combined_score", 0.0)),
        ),
    )
